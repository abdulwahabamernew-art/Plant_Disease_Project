import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
from PIL import Image

def preprocess(image, size=(224,224), normalize=True):
    x = np.asarray(image.convert("RGB").resize(tuple(size), Image.Resampling.BILINEAR), dtype=np.float32)
    return x / 255.0 if normalize else x

def top3(p, names):
    ids = np.argsort(p)[::-1][:3]
    return [{"index":int(i), "class":names[i], "confidence":float(p[i])} for i in ids]

class Classifier:
    def __init__(self, cfg):
        from .config import HOME
        self.backend=cfg.get("classifier_backend","torch")
        os.environ["KERAS_BACKEND"]=self.backend
        os.environ["KERAS_HOME"]=str(HOME/"configs/keras")
        if self.backend=="tensorflow":
            os.environ.setdefault("TF_NUM_INTRAOP_THREADS","4")
            os.environ.setdefault("TF_NUM_INTEROP_THREADS","2")
        import keras
        if keras.backend.backend()!=self.backend:
            raise RuntimeError("Switching Keras backends requires a fresh process")
        if self.backend=="torch":
            import torch
            torch.set_num_threads(4)
        self.cfg = cfg
        self.disease = keras.models.load_model(cfg["models"]["disease"], compile=False)
        self.gate = keras.models.load_model(cfg["models"]["gate"], compile=False)
        if tuple(self.disease.input_shape[1:]) != (224,224,3) or tuple(self.gate.input_shape[1:]) != (224,224,3):
            raise ValueError("Unexpected model input shape")
        if cfg["gate"].get("positive_class") != "plant":
            raise ValueError("Gate positive class must be plant")
        if self.disease.output_shape[-1] != len(cfg["classes"]) or self.gate.output_shape[-1] != 1:
            raise ValueError("Unexpected model output shape")
    def run(self,model,x):
        if self.backend=="torch":
            import torch
            with torch.inference_mode():return model(x,training=False).detach().cpu().numpy()
        return model(x,training=False).numpy()
    def plant(self, image):
        x = preprocess(image, self.cfg["gate"]["image_size"], False)
        p = float(self.run(self.gate,x[None]).reshape(-1)[0])
        if not np.isfinite(p) or not 0 <= p <= 1: raise ValueError("Invalid plant probability")
        g = self.cfg["gate"]
        label = "not_plant" if p < g["uncertain_lower_threshold"] else ("plant" if p >= g["plant_threshold"] else "uncertain")
        return {"label":label, "plant_probability":p}
    def predict(self, images):
        chunks=[]
        for start in range(0,len(images),16):
            x=np.stack([preprocess(im) for im in images[start:start+16]])
            chunks.append(self.run(self.disease,x))
        p=np.concatenate(chunks)
        if not np.all(np.isfinite(p)) or np.any(p < 0) or not np.allclose(p.sum(axis=1),1,atol=1e-3):
            raise ValueError("Classifier did not return probabilities")
        return p
