# Visual Prompting in LLMs for Emotion Recognition (Set-of-Vision) - modular project

Badagoni Sri Sahasra (B24DS003), IIT Bhilai. Reproduces and extends Zhang et al., EMNLP 2024.

## Structure
```
src/                 reusable modules (config, data, datasets, composites, detection, labeling, prompts, models, evaluation, plots, utils)
phase 1 code/        own-photo pipeline, one script per stage (01 preprocess -> 05 evaluate)
phase 2 code/        FER-2013 + WIDER FACE pipeline, one script per stage (01 -> 07)
notebooks/           the same stages as Colab notebooks (01-09)
results/             raw model outputs + metrics  (phase 1 = real saved Qwen2-VL-7B results)
analysis_folders/    figures, tables, summary.md per experiment
data/                put raw data here (fer2013/, wider_face/, raw/ for your own photos)
```
## Datasets and why
- **FER-2013**: 7-class emotion labels (48x48 faces) -> emotion ground truth. Labels are noisy (~65% human agreement); report this.
- **WIDER FACE**: crowded photos with face boxes -> (a) realistic backgrounds, (b) test whether the face detector SoV relies on finds all faces (recall/precision/face-count).
- Phase-2 test images = FER faces placed on a (blurred WIDER) canvas, 2-8 per image. Boxes + labels are known exactly, so no manual labelling.

## Run order (Phase 2)
```
pip install -r requirements.txt
python "phase 2 code/01_prepare_fer.py" --fer data/fer2013 --split test --per_class 60
python "phase 2 code/02_prepare_wider.py" --split_txt data/wider_face/wider_face_split/wider_face_val_bbx_gt.txt --images data/wider_face/WIDER_val/images --n 100
python "phase 2 code/03_wider_detector_eval.py"            # needs retina-face
python "phase 2 code/04_build_group_images.py" --n_images 80
python "phase 2 code/05_run_model.py" --model qwen          # GPU (Colab T4); then --model llava
python "phase 2 code/06_evaluate.py" --model qwen
python "phase 2 code/07_compare_models.py" --models qwen,llava
```
Smoke test with no GPU: use `--model mock` (random answers - never report those numbers).

## Status (honest)
- Verified here with synthetic data: steps 01, 02 parser, 04, 05 (mock), 06, 07 and rendering of all 10 variants.
- NOT run here (need GPU / your real data): real Qwen2-VL / LLaVA inference on phase 2, detector evaluation on WIDER. Fill the numbers from your runs into REPORT.md.
- Phase 1 numbers (Qwen2-VL-7B, 17 photos) are real and in results/phase1_own_photos_qwen2vl7b.
