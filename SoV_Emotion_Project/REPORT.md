# Visual prompting (Set-of-Vision) for zero-shot emotion recognition - final report

*Badagoni Sri Sahasra, B24DS003 (individual project). Reproduction and extension of Zhang et al., EMNLP 2024.*

> **Status of this report.** Sections 1-5 and 8 are complete and use only numbers from the saved run
> (Qwen2-VL-7B-Instruct, 4-bit, greedy decoding; `results/Qwen_Qwen2-VL-7B-Instruct/`). Sections marked **[PHASE 3 - TO FILL]** need the
> new photos and the Colab runs described in `PHASE3_GUIDE.md`; they contain the exact tables to fill and what each outcome would mean.
> Delete this note when everything is filled.

## 1. Problem statement
Vision-language models read images well overall, but in real photos with several people, cluttered backgrounds and small faces they often look at the
wrong face when asked for an emotion. Set-of-Vision (SoV) prompting adds spatial cues to the image - bounding boxes, face numbers and facial
landmarks - but uses them as one fixed bundle. Which cue actually helps, and by how much, had not been isolated.

## 2. Approach
No model is trained. We keep the image, the face detections, the question and the model settings identical and change **only what is drawn on
the image**. Ten visual prompts are compared: plain image (baseline), RedCircle-style, ReCLIP-style, boxes, boxes + numbers, full SoV (boxes + numbers +
landmarks), numbers only, landmarks only, and (Phase 3) full SoV / landmarks only with much smaller dots.

**Standardized prompt protocol.** One fixed question for every variant: the 7 allowed labels (angry, disgust, fear, happy, sad, surprise, neutral),
exactly one label per face, the number tag as the explicit face id where one is drawn (otherwise "list faces left to right"), and a JSON answer
`{"num_faces": n, "faces": [{"id": k, "emotion": "<label>"}]}`. The answer parser maps free text to the 7 labels; anything else counts as wrong.

## 3. Pipeline and code structure
Faces are detected with RetinaFace; faces under 32 px are ignored; overlapping boxes are removed with the paper's Algorithm 1 (keep the larger face,
overlap threshold 0.5); faces are numbered left to right; landmarks (eyes, nose, mouth corners) come from the detector.
Ground-truth labels: DeepFace first guess, then manual review. Each stage is its own notebook backed by reusable modules in `src/`
(see README): 01 data preparation -> 02 detection -> 03 labels -> 04 visual prompts -> 05 inference -> 06 evaluation -> 07 analysis, plus
08 landmark ablation and 09 second model for Phase 3. The rebuilt evaluation code reproduces the saved metrics exactly (difference < 1e-15).

## 4. Setup of the main study
- **Model:** Qwen2-VL-7B-Instruct, run locally, 4-bit, greedy decoding. The paper used GPT-4V / GPT-4o, so absolute numbers are not comparable.
- **Data:** 18 photos uploaded, 17 used (1 without a detected face), 77 faces: Easy 4 images / 11 faces, Medium 12 / 57, Hard 1 / 9.
  The paper used 119 images / 459 faces. Emotion distribution: neutral 27, happy 24, sad 15, angry 6, fear 2, surprise 2, disgust 1.
- **Labels:** DeepFace guess, manually reviewed (4 of 77 labels changed, 5.2%), one annotator.
- **Metrics:** accuracy (per face), macro precision / recall / F1 over the emotions present, face-count accuracy (per image),
  per-class recall, and 95% bootstrap intervals for the difference vs. the plain image (resampling images).
  The paper's "Recall@1" is not defined, so macro recall is our interpretation.
- **Matching:** numbered variants match answer id *k* to ground-truth face *k*; the others match the *k*-th answer to the *k*-th face from the left.
  Missing, invalid or unparsable answers count as wrong. 136 of 136 answers parsed.

## 5. Results of the main study (all 77 faces; always-"neutral" baseline = 35.1%)
| Variant | Accuracy | Macro recall | Macro precision | Macro F1 | Face-count acc. |
|---|---|---|---|---|---|
| V0 Plain (baseline) | 46.8 | 28.7 | 33.1 | 28.9 | 94.1 |
| V1 RedCircle-style | **53.2** | 33.7 | 44.4 | 34.8 | 88.2 |
| V2 ReCLIP-style | 42.9 | 27.0 | 33.8 | 27.6 | 88.2 |
| V3 Boxes | 42.9 | 27.0 | 34.3 | 27.9 | 88.2 |
| V4 Boxes + numbers | 49.4 | 32.0 | 45.1 | 31.5 | **100** |
| V5 Full SoV | 41.6 | 17.5 | 14.2 | 13.8 | **100** |
| V6 Numbers only | 48.1 | 31.4 | 39.9 | 29.0 | **100** |
| V7 Landmarks only | 35.1 | 14.9 | 10.9 | 12.1 | 94.1 |

**Difference vs plain (accuracy, 95% bootstrap CI):** V1 +6.5 pts (+1.1 to +12.9, excludes 0); V2 -3.9 (-9.6 to +1.4); V3 -3.9 (-9.7 to +1.3);
V4 +2.6 (-4.3 to +9.5); V5 -5.2 (-19.3 to +6.0); V6 +1.3 (-7.9 to +11.0); V7 -11.7 (-26.0 to -1.1, excludes 0).

**Ablation (plain -> box -> box+number -> full SoV):** ours 46.8 -> 42.9 -> 49.4 -> 41.6; paper 44.4 -> 44.7 -> 51.6 -> 55.3.

### Analysis
1. **The model is not guessing.** Plain accuracy (46.8%) is 11.7 points above the majority baseline; it is close to the paper's 44.4%
   (different model and data, so only loosely comparable).
2. **Number tags fix counting, not emotion.** Every variant with number tags counted the faces correctly on all images (100% vs 94.1% for plain;
   the gap is one image, the Hard one). Boxes or circles without numbers did not help counting (88.2%). The accuracy changes for V4 / V5 / V6
   (+2.6 / -5.2 / +1.3 pts) are not significant.
3. **Box + number is the best SoV-style cue, numbers alone do about as well** (49.4% vs 48.1%); boxes alone are -3.9 pts. None of these differences is statistically clear.
4. **Landmarks hurt, contradicting the paper.** Full SoV (41.6%) is below plain and landmarks only (V7) is significantly worse (-11.7 pts). In both the model
   says "neutral" for most faces (V5 82%, V7 70%) and happy recall drops from 0.75 (plain) to 0.33 (V5) and 0.38 (V7). V7's accuracy equals the always-neutral
   baseline and V5 is flagged as collapsed, so V5's 41.6% is not real recognition. Possible reasons (tested in Phase 3): the dots cover eyes and mouth
   on small faces, or the model does not use dot markers.
5. **RedCircle was the only significant gain (+6.5 pts)** but lowered face-count accuracy (88.2%). The paper ranks it below SoV; we see the opposite.
6. **Per class:** only happy (0.67-0.79 without landmarks, 0.33-0.38 with), neutral (0.56-0.89; highest for V5 only because it says "neutral" for almost everything),
   angry (0-0.33) and surprise (0-0.50) are recognised. Disgust (1 face), fear (2) and sad (15 faces, recall 0-0.07) are almost never recognised; classes with 1-2 faces cannot support conclusions.
7. **By difficulty:** Easy has 11 faces and Hard 9 faces from one image, so those columns are not interpretable.

## 6. Phase 3 - extensions
### 6.1 Bigger dataset and label review  **[PHASE 3 - TO FILL]**
Photos added: ___ new (total ___ images, ___ faces); Easy ___ / Medium ___ / Hard ___ images; emotion counts: angry ___, disgust ___, fear ___, happy ___, sad ___, surprise ___, neutral ___.
Labels re-checked face by face: ___ labels changed (___%). Re-run of the main study on the larger set (notebooks 05-07): fill the Section 5 table again and say which conclusions changed.

### 6.2 Landmark ablation (small dots)  **[PHASE 3 - TO FILL]**  (`notebook 08`)
| Variant | Accuracy % | Macro F1 % | Most common answer (share %) | Happy recall % |
|---|---|---|---|---|
| V0 Plain | | | | |
| V4 Boxes + numbers | | | | |
| V5 Full SoV (big dots) | | | | |
| V8 Full SoV, small dots | | | | |
| V7 Landmarks only (big dots) | | | | |
| V9 Landmarks only, small dots | | | | |

*If V8 / V9 recover* (higher accuracy than V5 / V7, fewer "neutral" answers, happy recall back near plain): the big dots were hiding the faces.
*If they do not:* dot size is not the cause and the model does not benefit from landmark dots. Report the confidence intervals either way.

### 6.3 Second model: LLaVA-1.5-7B  **[PHASE 3 - TO FILL]**  (`notebook 09`)
| Variant | Qwen2-VL-7B acc. % | LLaVA-1.5-7B acc. % | LLaVA face-count % |
|---|---|---|---|
| V0 ... V7 | | | |

Questions to answer: do the same patterns hold (number tags help counting; landmarks do not help accuracy)? Does LLaVA collapse to one answer?
Note that LLaVA-1.5 resizes images to 336x336, so small faces in crowded photos lose detail.

### 6.4 Standardized prompt protocol
Implemented as described in Section 2 (`src/prompts.py`): explicit face id, constrained output labels, fixed JSON format, one shared matching rule.
The main-study result that supports it is the face-count finding: with number tags the model identified the right faces on every image.

## 7. Comparison with the paper
Baseline and Box+Number behave similarly to the paper (46.8 vs 44.4; 49.4 vs 51.6). Full SoV does not (41.6 vs 55.3) and RedCircle beats SoV, so the study
**partly reproduces** the paper: numbered boxes help counting and slightly help accuracy, but the benefit of facial landmarks is not reproduced.
**[PHASE 3 - TO FILL]** after 6.2 / 6.3: say whether the landmark result depends on dot size or on the model.

## 8. Limitations
- Very small dataset (17 images, 77 faces in the main study; Hard has 1 image); most differences are inside the confidence intervals.
- Ground truth is DeepFace's output with light manual changes from a single annotator.
- Open models in 4-bit, one run each with greedy decoding, not GPT-4V.
- Matching differs between groups (numbered variants by tag, others by position), which can slightly favour numbered variants.
- V1/V2 omit the crop step of the original methods; "recall" is macro recall; face-count accuracy and bootstrap intervals are additions of this project.

## 9. Conclusion (main study; update after Phase 3)
Number tags make the model's face counting correct on every image (94.1% -> 100%, a one-image difference) and numbered boxes give a small, non-significant
accuracy gain, consistent with the direction of the paper. Facial-landmark dots did not help this model and reduced accuracy. With this little data the safe claim is:
*numbered markers help grounding and counting; landmarks need a larger dataset, smaller dots and another model to be judged.*

## References
Zhang, Wang, Zhang, Niu, Caldwell, Gedeon, Liu, Qin (2024). *Visual Prompting in LLMs for Enhancing Emotion Recognition.* EMNLP 2024, pp. 4484-4499.
