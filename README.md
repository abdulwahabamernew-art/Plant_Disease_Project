# Plant Disease Project

A source-only computer-vision pipeline for plant-disease classification with a separate plant/not-plant gate and optional leaf detection. The project is designed to reject obvious non-plant images before running the 38-class disease classifier.

> **Repository scope:** This GitHub repository contains safe source code, tests, dependency files, and documentation only. Private datasets, trained weights, runtime caches, generated reports, notebooks, Android build artifacts, and signing material are intentionally excluded.

## Highlights

- 38-class plant-disease classification using a Keras-compatible backend.
- A plant/not-plant gate with `not_plant`, `uncertain`, and `plant` outcomes.
- Optional YOLO-based leaf detection before per-leaf classification.
- Configurable confidence thresholds, crop padding, and raw/crop fusion.
- JSON metrics and annotated-image output kept inside the local runtime output directory.
- Tests covering core pipeline behavior, image handling, output scope, and UI presentation.

## Architecture

```text
input image
    │
    ▼
plant/not-plant gate ──► reject or mark uncertain
    │
    ▼
leaf detector
    │
    ▼
per-leaf crops ──► 38-class disease classifier
    │
    ▼
JSON metrics + optional annotated image
```

The main implementation is under [`detection_v2/src`](detection_v2/src). The pipeline loads the gate and disease models through [`src/config.py`](detection_v2/src/config.py), and the detector worker is isolated in a subprocess so the inference components can be controlled independently.

## Repository layout

```text
.
├── detection_v2/
│   ├── src/                  # inference pipeline and model adapters
│   ├── tests/                # project tests
│   ├── scripts/              # safe diagnostic/test helpers only
│   ├── config/               # instructions for private runtime config
│   ├── requirements*.txt     # dependency variants
│   └── README.md             # original project notes
├── .gitignore                # blocks secrets, datasets, weights, and outputs
├── SECURITY.md               # security and secret-handling guidance
└── .env.example              # empty local-settings template
```

## Results and limitations

The source project reports a PlantVillage test accuracy of **86.17%** for its MobileNetV2 transfer-learning model and **29.29%** for its simple CNN baseline. These figures are dataset-specific and should not be interpreted as real-world phone-camera accuracy. PlantDoc coverage in the source Drive project is a partial subset, so external validation remains necessary.

The detector thresholds are initial operating values, not calibrated guarantees:

- Plant probability below `0.40`: reject as not a plant.
- `0.40` to below `0.80`: return an uncertain result.
- `0.80` or higher: allow disease classification.

Validate thresholds using representative images from the intended deployment environment, with particular attention to false acceptance of non-plant images and false rejection of leaves.

## Local setup

The repository is intentionally not self-contained because model weights and private runtime configuration are not committed.

1. Create a Python environment and install one of the dependency sets in [`detection_v2/`](detection_v2/):

   ```bash
   python -m pip install -r detection_v2/requirements.txt
   # or choose requirements-tensorflow.txt / requirements-ui.txt as appropriate
   ```

2. Obtain the required runtime artifacts through a secure, private channel. See [`detection_v2/config/README.md`](detection_v2/config/README.md).
3. Place the private model files and configuration at the paths expected by `detection_v2/src/config.py`.
4. Run the tests after the local runtime is configured:

   ```bash
   pytest detection_v2/tests
   ```

Do not add private artifacts to this repository just to make local setup easier. Keep them in a local ignored directory or a separate access-controlled storage location.

## Continuous integration

GitHub Actions runs automatically on every `push` and Pull Request. The workflow
uses read-only repository permissions, compiles the Python source, and runs the
image-orientation tests that do not require private model weights or datasets.
Model-backed integration tests remain local until a separate secure test-artifact
strategy is defined.

## Security and privacy

Never commit any of the following:

- API keys, OAuth tokens, passwords, or service-account files.
- Android keystores, private signing keys, or signing passwords.
- Private datasets, user images, model credentials, or local cache files.
- Trained model weights unless they have been explicitly reviewed and are suitable for distribution.

The repository `.gitignore` blocks common secret, model, dataset, output, and Android artifact patterns. Review `git diff --cached` before every push. If a secret is ever exposed, revoke or rotate it immediately; removing it in a later commit is not sufficient. See [`SECURITY.md`](SECURITY.md).

## License

No license has been added yet. Add the appropriate license and attribution information before distributing the code publicly.
