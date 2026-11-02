from pathlib import Path
import hashlib,json,math,time,uuid
import numpy as np
from PIL import Image,ImageOps
from .config import HOME,load_config
from .classifier import Classifier,top3
from .detector import Detector
from .visualization import annotate
from .image_io import load_rgb

def crop_box(box,size,padding):
    x1,y1,x2,y2=map(float,box);w,h=size
    if padding < 0 or x2 <= x1 or y2 <= y1: raise ValueError("Invalid crop box/padding")
    dx=(x2-x1)*padding;dy=(y2-y1)*padding
    b=[max(0,math.floor(x1-dx)),max(0,math.floor(y1-dy)),min(w,math.ceil(x2+dx)),min(h,math.ceil(y2+dy))]
    if b[2]<=b[0] or b[3]<=b[1]:raise ValueError("Box lies outside image")
    return b

def fusion(raw,crop,threshold,alpha):
    return raw.copy() if raw.max()>=threshold else alpha*raw+(1-alpha)*crop

def checked_output_root(output_dir=None):
    root=Path(output_dir).resolve() if output_dir is not None else HOME/"outputs"
    if not root.is_relative_to(HOME.resolve()):
        raise ValueError("New pipeline outputs must remain inside detection_v2")
    return root

class Pipeline:
    def __init__(self, padding=None, imgsz=None, backend=None):
        start=time.perf_counter();self.cfg=load_config()
        if backend is not None:self.cfg["classifier_backend"]=backend
        if padding is not None:self.cfg["padding"]=padding
        if imgsz is not None:self.cfg["detector"]["imgsz"]=imgsz
        if self.cfg["padding"] < 0:raise ValueError("Padding must be nonnegative")
        self.classifier=Classifier(self.cfg);self.detector=Detector(self.cfg)
        self.load_seconds=time.perf_counter()-start
        self.run=uuid.uuid4().hex[:12]
    def close(self):self.detector.close()
    def predict(self,path,save=True,paddings=None,output_dir=None):
        start=time.perf_counter();path=Path(path).resolve()
        image=load_rgb(path)
        result={"input":str(path),"image_dimensions":list(image.size),"settings":self.cfg,"model_load_seconds":self.load_seconds,"detector_metadata":self.detector.metadata,"leaves":[],"review_flags":[],"timing":{}}
        t=time.perf_counter();result["plant_gate"]=self.classifier.plant(image);result["timing"]["plant_gate_seconds"]=time.perf_counter()-t
        if result["plant_gate"]["label"]=="not_plant":
            result["status"]="gate_rejected";result["review_flags"].append("gate_rejected_review_for_false_negative")
            result["raw"]=None;result["timing"].update(yolo_seconds=0,classifier_seconds=0)
        else:
            result["status"]="processed"
            if result["plant_gate"]["label"]=="uncertain":result["review_flags"].append("uncertain_gate")
            detected=self.detector.predict(path);result["timing"]["yolo_seconds"]=detected["seconds"];result["yolo_speed_ms"]=detected["speed_ms"]
            pads=list(dict.fromkeys([self.cfg["padding"],*(paddings or [])]))
            result["evaluated_paddings"]=pads
            views=[image];meta=[]
            for b in detected["boxes"]:
                crops=[]
                for pad in pads:
                    cb=crop_box(b["bbox"],image.size,pad);views.append(image.crop(cb));crops.append((pad,cb,len(views)-1))
                meta.append((b,crops))
            result["classification_views"]=len(views)
            t=time.perf_counter();probs=self.classifier.predict(views);result["timing"]["classifier_seconds"]=time.perf_counter()-t
            def prediction(p):return {"top3":top3(p,self.cfg["classes"]),"probabilities":p.tolist()}
            result["raw"]=prediction(probs[0])
            for i,(box,crops) in enumerate(meta,1):
                variants=[]
                for pad,cb,index in crops:
                    variants.append({"padding":pad,"crop_bbox":cb,**prediction(probs[index])})
                primary=variants[0];cp=np.array(primary["probabilities"])
                f=self.cfg["fusion"];fp=fusion(probs[0],cp,f["raw_threshold"],f["raw_weight"])
                flags=[]
                if primary["top3"][0]["confidence"]<self.cfg["low_confidence"]:flags.append("low_confidence")
                if primary["top3"][0]["index"]!=result["raw"]["top3"][0]["index"]:flags.append("raw_crop_disagreement")
                x1,y1,x2,y2=box["bbox"]
                if (x2-x1)*(y2-y1)/(image.width*image.height)<0.01:flags.append("small_object_candidate")
                if primary["top3"][0]["class"].startswith("Tomato") and not result["raw"]["top3"][0]["class"].startswith("Tomato"):flags.append("tomato_shift_candidate")
                result["leaves"].append({"id":i,**box,"crop":primary,"padding_variants":variants,"experimental_fusion":prediction(fp),"review_flags":flags,"diagnosis_status":"requires_per_leaf_ground_truth"})
            if not meta:result["review_flags"].append("no_detections_requires_visual_review")
        result["detected_leaves"]=len(result["leaves"])
        result["timing"]["pipeline_seconds"]=time.perf_counter()-start
        if save:
            out_root=checked_output_root(output_dir)
            for sub in ["annotated","crops","metrics","failed","low_confidence"]:(out_root/sub).mkdir(parents=True,exist_ok=True)
            token=f"{path.stem}_{hashlib.sha256(str(path).encode()).hexdigest()[:8]}_{self.run}"
            dest=out_root/"annotated"/f"{token}.jpg"
            annotate(image,result).save(dest,quality=92)
            result["annotated_image"]=str(dest)
            for leaf in result["leaves"]:
                for j,var in enumerate(leaf["padding_variants"]):
                    crop=out_root/"crops"/f"{token}_leaf{leaf['id']}_pad{j}.jpg"
                    image.crop(var["crop_bbox"]).save(crop,quality=92);var["saved_crop"]=str(crop)
            if result["status"]=="gate_rejected" or not result["leaves"]:
                image.save(out_root/"failed"/f"{token}.jpg")
            if result["plant_gate"]["label"]=="uncertain" or any("low_confidence" in l["review_flags"] for l in result["leaves"]):
                image.save(out_root/"low_confidence"/f"{token}.jpg")
            result["timing"]["total_with_output_seconds"]=time.perf_counter()-start
            out=out_root/"metrics"/f"{token}.json";out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
            result["json_path"]=str(out)
        return result
