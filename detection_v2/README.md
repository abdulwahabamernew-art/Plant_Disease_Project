# Plant Disease Classification Project

TensorFlow/Keras project for 38-class plant disease classification, with a separate plant/not-plant gate before disease prediction.

## Current disease-model results

| Model | PlantVillage test accuracy |
|---|---:|
| Simple CNN | 29.29% |
| MobileNetV2 transfer learning | 86.17% |

These results use the reserved PlantVillage test set of 10,948 images. The plant detector is a separate model and has not yet been trained in this Drive copy.

## Canonical execution order

For a complete rebuild, run:

1. `01_Dataset_Collection_REPAIRED.ipynb`
2. `02_Preprocessing_and_Training_Splits_MEMORY_SAFE.ipynb`
3. `03_Model_Training_CNN_vs_TransferLearning.ipynb`
4. `05_Model_Evaluation_and_Comparison.ipynb`
5. `07_Plant_Detector_Data_Preparation.ipynb`
6. `08_Train_Plant_Detector.ipynb`
7. `09_Enhanced_Gradio_Interface.ipynb`

The disease model files are already present, so ordinary inference does not require rerunning stages 01-05. After the detector is trained and its config exists, run stage 09.

## Avoiding duplicate training

`03_Baseline_CNN_Training.ipynb` and `04_Transfer_Learning_MobileNetV2.ipynb` are optional model-specific alternatives. The combined stage 03 trains both models and is the only stage 03 used in the canonical order. Do not run the alternatives as part of that same workflow: they retrain and overwrite their corresponding model files.

`06_Gradio_Plant_Disease_Interface.ipynb` is a legacy disease-only interface. It bypasses the plant/not-plant detector and can label unrelated images as a disease. Use stage 09 or the source-controlled `app.py` interface as the final interface.

## Local Gradio interface

The repository includes a source-controlled interface at [`app.py`](app.py). It keeps the plant gate visible, separates detector-only diagnostics from disease classification, and exposes the annotated image, per-leaf table, crops, and JSON evidence.

From the repository root:

```bash
python -m pip install -r detection_v2/requirements-ui.txt
python detection_v2/app.py
```

The UI is intentionally evidence-first. It labels uncertain or low-confidence results for review and never describes model confidence as diagnostic accuracy. The detector-only button is useful for checking the gate and leaf detector without running disease classification.

## Project structure

```text
Plant_Disease_Project/
├── detection_v2/
│   ├── app.py                 # source-controlled Gradio interface
│   ├── src/
│   ├── tests/
│   ├── scripts/
│   ├── config/
│   └── requirements*.txt
├── notebooks/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── plant_detector/
├── models/
├── config/
├── results/
└── reports/
```

The detector preparation notebook reads both the class folders under existing `train`, `validation`, and `test` directories and the `incoming` folders. Existing split assignments are preserved. Incoming images are deduplicated against those fixed splits, then split 70/15/15. The manifest, not copied split images, is the source used by stage 08.

Stage 07 can collect up to 5,000 starter images per class automatically. If using your own images only, set `AUTO_COLLECT = False` before running it. Review the dataset summary and manifest before training.

## Detector decision policy

- Plant probability below 0.40: reject as not a plant.
- 0.40 to below 0.80: return an uncertain result.
- At least 0.80: pass the image to the 38-class disease model.

These are initial thresholds. Validate them with external phone-camera images, prioritizing low false acceptance of non-plant images.

## Data notes

- PlantVillage: 54,304 images; 43,356 training and 10,948 reserved test images.
- PlantDoc in this Drive copy: 845 images, a partial subset of the original dataset.
- The 86.17% figure is PlantVillage test accuracy, not PlantDoc cross-dataset accuracy.
