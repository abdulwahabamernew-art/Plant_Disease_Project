import json, os, subprocess, sys
from pathlib import Path
from .config import HOME
class Detector:
    def __init__(self, cfg):
        self.cfg=cfg
        env=os.environ.copy()
        (HOME / "configs/ultralytics/Ultralytics").mkdir(parents=True,exist_ok=True)
        env["YOLO_CONFIG_DIR"]=str(HOME / "configs/ultralytics")
        env["PYTHONDONTWRITEBYTECODE"]="1"
        self.process=subprocess.Popen([sys.executable,str(HOME/"src/detector_worker.py"),cfg["models"]["detector"]],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding="utf-8",env=env,cwd=str(HOME))
        ready=self.process.stdout.readline()
        data=json.loads(ready) if ready else {}
        if not data.get("ready"):
            self.close(); raise RuntimeError("Detector worker failed to start")
        self.metadata=data["metadata"]
    def predict(self,path,settings=None):
        payload={"path":str(Path(path).resolve()),"settings":settings or self.cfg["detector"]}
        self.process.stdin.write(json.dumps(payload)+"\n"); self.process.stdin.flush()
        line=self.process.stdout.readline()
        if not line: raise RuntimeError("Detector worker stopped unexpectedly")
        result=json.loads(line)
        if "error" in result: raise RuntimeError(result["error"])
        return result
    def close(self):
        if getattr(self,"process",None):
            if self.process.poll() is None:
                self.process.stdin.close()
                try:self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:self.process.terminate();self.process.wait(timeout=10)
    def __del__(self):
        self.close()
