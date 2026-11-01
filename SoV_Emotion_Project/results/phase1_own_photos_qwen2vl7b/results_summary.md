# Results summary (Qwen_Qwen2-VL-7B-Instruct)

Images evaluated: 17; faces: 77.

## Overall
- V0 Plain (baseline): accuracy 46.8%, macro recall 28.7%, F1 28.9%, face-count accuracy 94.1%
- V1 RedCircle-style (circle + blur): accuracy 53.2%, macro recall 33.7%, F1 34.8%, face-count accuracy 88.2%
- V2 ReCLIP-style (box + blur): accuracy 42.9%, macro recall 27.0%, F1 27.6%, face-count accuracy 88.2%
- V3 Boxes only: accuracy 42.9%, macro recall 27.0%, F1 27.9%, face-count accuracy 88.2%
- V4 Boxes + numbers: accuracy 49.4%, macro recall 32.0%, F1 31.5%, face-count accuracy 100.0%
- V5 Full SoV (boxes+numbers+landmarks): accuracy 41.6%, macro recall 17.5%, F1 13.8%, face-count accuracy 100.0%
- V6 Numbers only: accuracy 48.1%, macro recall 31.4%, F1 29.0%, face-count accuracy 100.0%
- V7 Landmarks only: accuracy 35.1%, macro recall 14.9%, F1 12.1%, face-count accuracy 94.1%

## Difference vs plain image (accuracy, percentage points, 95% bootstrap CI)
- V1 RedCircle-style (circle + blur): +6.5 (CI +1.1 to +12.9) - interval excludes 0
- V2 ReCLIP-style (box + blur): -3.9 (CI -9.6 to +1.4) - interval includes 0 (not clearly different)
- V3 Boxes only: -3.9 (CI -9.7 to +1.3) - interval includes 0 (not clearly different)
- V4 Boxes + numbers: +2.6 (CI -4.3 to +9.5) - interval includes 0 (not clearly different)
- V5 Full SoV (boxes+numbers+landmarks): -5.2 (CI -19.3 to +6.0) - interval includes 0 (not clearly different)
- V6 Numbers only: +1.3 (CI -7.9 to +11.0) - interval includes 0 (not clearly different)
- V7 Landmarks only: -11.7 (CI -26.0 to -1.1) - interval excludes 0

Highest overall accuracy: V1 RedCircle-style (circle + blur) (53.2%).
- Majority-class baseline (always 'neutral'): accuracy 35.1%.
- WARNING: these variants gave one answer for >80% of faces, so their accuracy is not meaningful: V5 Full SoV (boxes+numbers+landmarks)