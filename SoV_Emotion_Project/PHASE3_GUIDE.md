# Phase 3 guide: what to collect, what to run

Phase 3 = the future work of the main study: **(A)** a bigger dataset, **(B)** a careful label review, **(C)** test why landmarks hurt
(smaller dots), **(D)** a second model (LLaVA-1.5-7B), **(E)** the standardized prompt protocol and final report.
(E) is already written in code/README; the rest needs your photos and a Colab GPU.

## Step 1 - collect more photos (about 1 hour)
Goal: **40+ images in total** (you have 17, so at least 23 new), spread across difficulties:

| Difficulty (faces per image) | Have now | Target |
|---|---|---|
| Easy (1-3) | 4 | 12+ |
| Medium (4-7) | 12 | 16+ |
| Hard (8+) | 1 | **8-10** |

Rules of thumb
- Multi-person photos with **clearly visible, front-facing faces** (>= ~40 px wide). Crowds, groups, events, news photos.
- Also look for the **emotions you are missing**: your labels are neutral 27, happy 24, sad 15, angry 6, fear 2, surprise 2, disgust 1.
  Add photos with angry, fearful, surprised and disgusted faces, otherwise those classes stay untestable.
- No duplicates or near-duplicates; one image per scene.
- **Name the new files `new_01.jpg, new_02.jpg, ...`** (do not reuse `image1.jpg`...). The saved answers are keyed by file name; a clash would
  make the notebook skip a new photo because an old answer with the same name exists.
- Put the new photos in one zip (e.g. `new_photos.zip`). Keep your original `ML PROJECT.zip` too - you need both.

## Step 2 - Colab setup (10 minutes)
1. Zip the repo folder, open https://colab.research.google.com, upload the notebooks later one at a time (File -> Upload notebook) or
   upload the repo zip to the Colab file panel and run `!unzip -q sov-emotion-recognition.zip && %cd sov-emotion-recognition`.
2. Runtime -> Change runtime type -> **T4 GPU**.
3. Open each notebook from the `notebooks/` folder; the first cell finds the repo automatically.

## Step 3 - run order
| # | Notebook | You do | Approx. time on a T4 |
|---|---|---|---|
| 1 | `01_data_preparation` | upload **both** zips (original + new photos) when asked | 2 min |
| 2 | `02_face_detection` | just run | 5-10 min |
| 3 | `03_labeling` | **review every face** in the pictures and fill `NEW_CORRECTIONS` (Step 4), set `REVIEWED = True` | 30-60 min of your time |
| 4 | `04_visual_prompts` | run; check the numbers/boxes sit on the right faces | 2 min |
| 5 | `05_model_inference` (`PROVIDER = "qwen"`) | run; only **new** images are queried, old answers are kept | ~20-40 min |
| 6 | `06_evaluation`, `07_analysis_and_report` | run (they also work on a laptop) | 2 min |
| 7 | `08_landmark_ablation` | run (adds V8/V9 answers) | ~15-30 min |
| 8 | `09_second_model_llava` | run (loads LLaVA, runs all 8 variants) | ~20-40 min |

Times are rough estimates, not measurements; the notebooks resume where they stopped if Colab disconnects.
Save results to Google Drive or download the `results/` folder after each long step - Colab files disappear when the runtime resets.

## Step 4 - how to review labels (the part that matters most)
In notebook 03 every face is drawn with a box, a number and DeepFace's guess (`3:sad`). For each face decide the real emotion.
Write the wrong ones into `NEW_CORRECTIONS`, e.g.
```python
NEW_CORRECTIONS = {"new_03.jpg": {1: "happy", 4: "angry"}, "new_07.jpg": {2: "ignore"}}   # "ignore" removes a wrongly detected face
```
Your earlier fixes (image10, image16, image18, image20) are loaded from `data/labels_corrections.json` automatically.
Also re-check the 15 faces labelled "sad" in the old photos - the model almost never agreed with them, so some may be DeepFace errors.
If you change old labels, put them in `NEW_CORRECTIONS` too (it overrides the saved ones).

## Step 5 - bring the results back
After step 3 (and the runs), download the `results/` and `data/` folders (zip) and send them to me. I will check the numbers,
fill the Phase 3 sections of `REPORT.md` (they are marked), update the README table, and repackage for GitHub.
