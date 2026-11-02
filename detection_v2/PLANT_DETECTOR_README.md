# Plant / Not Plant Gate

The detector is a binary gate that runs before the existing 38-class disease classifier. It does not replace or overwrite `models/mobilenetv2_best.keras`.

## Data folders and source priority

```text
data/plant_detector/
├── incoming/
│   ├── plant/
│   └── not_plant/
├── train/                 # optional manually assigned split
│   ├── plant/
│   └── not_plant/
├── validation/            # optional manually assigned split
│   ├── plant/
│   └── not_plant/
├── test/                  # optional manually assigned split
│   ├── plant/
│   └── not_plant/
└── plant_detector_manifest.csv
```

Stage 07 reads existing split folders and preserves their assignments. It then adds valid, unique images from `incoming`, assigning only those incoming images to train/validation/test using a 70/15/15 ratio. Exact duplicates are removed across all folders. Stage 08 reads the manifest; it does not read class folders directly.

If you want to use your own images without downloading the starter dataset, set `AUTO_COLLECT = False` in stage 07 and place labeled images in either the existing split folders or `incoming/plant` and `incoming/not_plant`.

## Run order

1. Run `07_Plant_Detector_Data_Preparation.ipynb` completely.
2. Review `data/plant_detector/plant_detector_manifest.csv` and `config/plant_detector_dataset_summary.json`.
3. Run `08_Train_Plant_Detector.ipynb` with a GPU runtime.
4. Run `09_Enhanced_Gradio_Interface.ipynb`.

The model is saved to `models/plant_detector/plant_detector_best.keras`; its runtime config is saved to `config/plant_detector_config.json`. The config template is not a trained-model config and is not used by stage 09.

## Dataset guidance

- Use diverse close leaf images for the plant class, including healthy and diseased leaves.
- Include hard negatives such as food, flowers, fruit-only images, balloons, people, vehicles, buildings, animals, indoor scenes, and empty frames.
- Keep manually assigned test images out of training folders.
- Review model errors and thresholds against external phone-camera images.

## Decision policy

- Below `0.40`: reject as not a plant.
- From `0.40` to below `0.80`: report uncertain.
- At or above `0.80`: run disease classification.

These initial thresholds should be tuned with priority given to reducing false acceptance of non-plant images.
