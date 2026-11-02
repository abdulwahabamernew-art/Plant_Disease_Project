import contextlib,json,sys,time
from pathlib import Path
from PIL import Image,ImageOps
from image_io import load_rgb

def reply(data):
    sys.stdout.write(json.dumps(data)+"\n");sys.stdout.flush()
with contextlib.redirect_stdout(sys.stderr):
    from ultralytics import YOLO
    model=YOLO(sys.argv[1])
    import yaml
    dataset=Path(__file__).resolve().parents[2]/"data/object_detection/plantdoc_leaf_v1/plantdoc_leaf_v1.yaml"
    names=yaml.safe_load(dataset.read_text(encoding="utf-8"))["names"]
    args=model.ckpt.get("train_args",{})
    if names!={0:"leaf"} or model.model.yaml.get("nc")!=1 or set(model.names)!={0}:
        raise ValueError("Detector/dataset is not the audited single-class leaf model")
    if model.names[0]=="item":
        if args.get("single_cls") is not True or Path(args.get("data","")).name!=dataset.name:
            raise ValueError("Generic item label lacks verified leaf training provenance")
    elif model.names[0]!="leaf":raise ValueError(f"Unexpected checkpoint class: {model.names}")
metadata={"checkpoint_names":model.names,"semantic_class_mapping":{0:"leaf"},"single_cls":args.get("single_cls"),"training_data":args.get("data"),"semantic_source":str(dataset)}
reply({"ready":True,"metadata":metadata})
for line in sys.stdin:
    try:
        request=json.loads(line)
        image=load_rgb(request["path"])
        start=time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            result=model.predict(image,device="cpu",verbose=False,save=False,**request["settings"])[0]
        boxes=[{"bbox":b.xyxy[0].tolist(),"yolo_confidence":float(b.conf[0]),"class_id":int(b.cls[0]),"class_name":"leaf","checkpoint_class_name":model.names[int(b.cls[0])]} for b in result.boxes]
        reply({"boxes":boxes,"seconds":time.perf_counter()-start,"speed_ms":result.speed,"image_dimensions":list(image.size)})
    except Exception as exc:reply({"error":f"{type(exc).__name__}: {exc}"})
