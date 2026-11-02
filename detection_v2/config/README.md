# Runtime configuration

Runtime model files and local configuration are intentionally not committed.
Obtain them privately from the project owner and place them under the paths
referenced by `detection_v2/src/config.py`.

Required private artifacts include:

- `config/class_names.json` (38 classifier labels)
- `config/plant_detector_config.json` (gate thresholds and image size)
- `detection_v2/configs/baseline.json` (non-secret runtime settings)
- model files under `models/` (Keras/YOLO weights)

Never commit API keys, OAuth credentials, signing keys, passwords, model
credentials, or private datasets. Use environment variables for secrets.
