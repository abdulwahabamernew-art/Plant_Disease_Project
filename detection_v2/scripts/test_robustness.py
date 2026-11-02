import argparse,io,json,sys,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageOps,ImageEnhance,ImageFilter,ImageDraw
from src.pipeline import Pipeline
from src.config import HOME
from src.image_io import load_rgb

def main():
    p=argparse.ArgumentParser();p.add_argument("image",type=Path);a=p.parse_args()
    original=load_rgb(a.image)
    dest=HOME/"outputs/robustness"/uuid.uuid4().hex[:12];dest.mkdir(parents=True)
    rng=np.random.default_rng(42);arr=np.asarray(original,dtype=np.float32)
    noisy=Image.fromarray(np.clip(arr+rng.normal(0,12,arr.shape),0,255).astype("uint8"))
    occluded=original.copy();d=ImageDraw.Draw(occluded);w,h=original.size;d.rectangle((w*.4,h*.3,w*.6,h*.7),fill=(90,80,60))
    shadow=arr.copy();shadow[:, :w//2]*=.4;shadow=Image.fromarray(shadow.astype("uint8"))
    # Horizontal motion blur via shifted copies, with edge padding (no wrap-around).
    padded=np.pad(arr,((0,0),(4,4),(0,0)),mode="edge")
    motion=Image.fromarray(np.mean([padded[:,i:i+w] for i in range(9)],axis=0).astype("uint8"))
    perspective=original.transform(original.size,Image.Transform.PERSPECTIVE,(1,.08,0,.04,1,0,.00015,.0001),Image.Resampling.BILINEAR)
    variants={"original":original,"dark":ImageEnhance.Brightness(original).enhance(.5),"bright":ImageEnhance.Brightness(original).enhance(1.5),"contrast":ImageEnhance.Contrast(original).enhance(.6),"shadow":shadow,"blur":original.filter(ImageFilter.GaussianBlur(2)),"motion_blur":motion,"noise":noisy,"rotation":original.rotate(15,resample=Image.Resampling.BILINEAR),"small_scale":original.resize((max(1,w//2),max(1,h//2))).resize((w,h)),"perspective":perspective,"occlusion":occluded}
    buffer=io.BytesIO();original.save(buffer,format="JPEG",quality=20);buffer.seek(0)
    with Image.open(buffer) as im:variants["jpeg20"]=im.convert("RGB")
    pipe=Pipeline();rows=[]
    try:
        for name,im in variants.items():
            path=dest/f"{name}.png";im.save(path)
            result=pipe.predict(path)
            rows.append({"variant":name,"gate":result["plant_gate"],"detections":result["detected_leaves"],"timing":result["timing"],"result":result["json_path"]})
            print(name,result["plant_gate"]["label"],result["detected_leaves"],flush=True)
    finally:pipe.close()
    (dest/"summary.json").write_text(json.dumps({"warning":"Synthetic sensitivity probe, not real-phone accuracy. Geometry and visibility change; no correctness inferred without reviewed variant-specific boxes and disease labels.","rows":rows},indent=2),encoding="utf-8")
    print(dest)
if __name__=="__main__":main()
