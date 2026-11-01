"""Stage 05: vision-language model providers and the experiment loop.

Providers: 'qwen' (Qwen2-VL-7B, local, 4-bit), 'llava' (LLaVA-1.5-7B, local, 4-bit),
'gemini' (API) and 'mock' (fake random answers, only to test the pipeline - never report them).
Every provider is called as provider(jpeg_bytes, prompt, rec=..., variant=...) and returns raw text.
"""
import io
import json
import os
import random
import time

from . import config, prompts, utils


class QuotaExhausted(Exception):
    pass


class FatalAPIError(Exception):
    pass


class MockProvider:
    """FAKE random answers - only to test that the pipeline runs. Never report these numbers."""
    def __call__(self, jpeg, prompt, rec=None, variant=None):
        rng = random.Random(f"{rec['image']}|{variant}")
        n = len(rec["faces"])
        return json.dumps({"num_faces": n, "faces": [{"id": i, "emotion": rng.choice(config.EMOTIONS)} for i in range(1, n + 1)]})


def _bnb_config(torch):
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")


class QwenProvider:
    """Local Qwen2-VL-7B-Instruct, greedy decoding, optional 4-bit quantisation (fits a free 15 GB T4)."""
    def __init__(self, model_name=config.MODEL_NAMES["qwen"], four_bit=True, max_pixels=1000 * 28 * 28):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        if not torch.cuda.is_available():
            raise FatalAPIError("No GPU found. In Colab use Runtime > Change runtime type > T4 GPU, then run again.")
        kw = {"device_map": "auto"}
        if four_bit:
            kw["quantization_config"] = _bnb_config(torch)
        else:
            kw["torch_dtype"] = torch.float16
        print("loading", model_name, "(4-bit)" if four_bit else "(fp16)")
        self.torch = torch
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, **kw).eval()
        self.processor = AutoProcessor.from_pretrained(model_name, min_pixels=256 * 28 * 28, max_pixels=max_pixels)

    def __call__(self, jpeg, prompt, rec=None, variant=None):
        from PIL import Image
        image = Image.open(io.BytesIO(jpeg)).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=900, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0]


class LlavaProvider:
    """Local LLaVA-1.5-7B (Phase 3 second model), greedy decoding, 4-bit.
    Note: LLaVA-1.5 resizes every image to 336x336, so small faces in crowded photos lose detail."""
    def __init__(self, model_name=config.MODEL_NAMES["llava"], four_bit=True):
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        if not torch.cuda.is_available():
            raise FatalAPIError("No GPU found. In Colab use Runtime > Change runtime type > T4 GPU, then run again.")
        kw = {"device_map": "auto"}
        if four_bit:
            kw["quantization_config"] = _bnb_config(torch)
        else:
            kw["torch_dtype"] = torch.float16
        print("loading", model_name, "(4-bit)" if four_bit else "(fp16)")
        self.torch = torch
        self.model = LlavaForConditionalGeneration.from_pretrained(model_name, **kw).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

    def __call__(self, jpeg, prompt, rec=None, variant=None):
        from PIL import Image
        image = Image.open(io.BytesIO(jpeg)).convert("RGB")
        conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.model.device, self.torch.float16)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=600, do_sample=False)
        return self.processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


class GeminiProvider:
    def __init__(self, model_name=config.MODEL_NAMES["gemini"]):
        from google import genai
        from google.genai import types
        try:
            from google.colab import userdata
            key = userdata.get("GEMINI_API_KEY")
        except Exception:
            key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise FatalAPIError("GEMINI_API_KEY not found. Add it in Colab Secrets and switch Notebook access on.")
        self.model_name, self.types = model_name, types
        self.client = genai.Client(api_key=key)

    def __call__(self, jpeg, prompt, rec=None, variant=None):
        t = self.types
        r = self.client.models.generate_content(
            model=self.model_name,
            contents=[t.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), prompt],
            config=t.GenerateContentConfig(temperature=0))
        return r.text or ""


_PROVIDERS = {"mock": MockProvider, "qwen": QwenProvider, "llava": LlavaProvider, "gemini": GeminiProvider}
_CACHE = {}


def get_provider(name):
    if name not in _CACHE:
        _CACHE[name] = _PROVIDERS[name]()
    return _CACHE[name]


def call_with_retry(provider, jpeg, prompt, rec=None, variant=None, tries=3):
    for attempt in range(tries):
        try:
            return provider(jpeg, prompt, rec=rec, variant=variant)
        except (QuotaExhausted, FatalAPIError):
            raise
        except Exception as e:
            msg = str(e)
            low = msg.lower()
            if "api key" in low or "permission" in low or "401" in msg or "403" in msg:
                raise FatalAPIError(msg[:200])
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                if "perday" in low.replace(" ", "").replace("_", "") or "per day" in low:
                    raise QuotaExhausted(msg[:200])
                time.sleep(20 * (attempt + 1))
                continue
            if attempt == tries - 1:
                raise
            time.sleep(2)
    raise RuntimeError("retries exhausted")


def round_robin(data):
    """Interleave Easy/Medium/Hard images so an interrupted run still covers every difficulty."""
    groups = {}
    for r in data:
        groups.setdefault(r["difficulty"], []).append(r)
    out = []
    for k in range(max((len(g) for g in groups.values()), default=0)):
        for g in groups.values():
            if k < len(g):
                out.append(g[k])
    return out


def load_done(raw_path):
    done = {}
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["image"], r["variant"])] = r["status"]
    return done


def run_experiment(data, provider_name, variants=config.VARIANTS, raw_path=None, limit_images=None,
                   max_requests=None, sleep=0.0):
    """Ask the model about every (image, variant) not answered yet; append raw answers to raw_outputs.jsonl.
    Safe to stop and re-run: finished pairs are skipped."""
    raw_path = raw_path or (config.results_dir(provider_name) / "raw_outputs.jsonl")
    provider = get_provider(provider_name)
    data = round_robin(data)
    if limit_images:
        data = data[:limit_images]
    done = load_done(raw_path)
    todo = [(r, v) for r in data for v in variants if done.get((r["image"], v)) != "ok"]
    print(f"images={len(data)} variants={len(variants)} total={len(data) * len(variants)} to do={len(todo)}")
    if max_requests:
        todo = todo[:max_requests]
    with open(raw_path, "a") as fh:
        for rec, variant in utils.progress(todo, desc=provider_name):
            jpeg = prompts.encode_jpeg(prompts.render_record(rec, variant))
            t0 = time.time()
            try:
                text = call_with_retry(provider, jpeg, prompts.build_prompt(variant), rec=rec, variant=variant)
                err = None
            except (QuotaExhausted, FatalAPIError) as e:
                print("\nSTOPPED:", e)
                break
            except Exception as e:
                text, err = "", f"{type(e).__name__}: {str(e)[:200]}"
            parsed = prompts.parse_answer(text)
            status = "ok" if parsed else ("api_error" if err else "parse_error")
            fh.write(json.dumps({"image": rec["image"], "variant": variant, "model": config.MODEL_NAMES[provider_name],
                                 "status": status, "raw_text": text, "parsed": parsed, "error": err,
                                 "seconds": round(time.time() - t0, 2), "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
            fh.flush()
            if provider_name != "mock":
                time.sleep(sleep)
    final = load_done(raw_path)
    print("status counts:", {s: list(final.values()).count(s) for s in set(final.values())}, "| file:", raw_path)
