from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[2]
HOME = ROOT / "detection_v2"
def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def load_config():
    cfg = read_json(HOME / "configs/baseline.json")
    cfg["classes"] = read_json(ROOT / "config/class_names.json")
    if isinstance(cfg["classes"], dict):
        cfg["classes"] = [cfg["classes"][str(i)] for i in range(len(cfg["classes"]))]
    cfg["gate"] = read_json(ROOT / "config/plant_detector_config.json")
    if len(cfg["classes"]) != 38:
        raise ValueError("Expected 38 classifier labels")
    for key, rel in cfg["models"].items():
        cfg["models"][key] = str(ROOT / rel)
        if not Path(cfg["models"][key]).is_file():
            raise FileNotFoundError(cfg["models"][key])
    if not 0 <= cfg["gate"]["uncertain_lower_threshold"] < cfg["gate"]["plant_threshold"] <= 1:
        raise ValueError("Invalid gate thresholds")
    return cfg
