import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.pipeline import Pipeline

def main():
    parser=argparse.ArgumentParser();parser.add_argument("image",type=Path);parser.add_argument("--padding",type=float);parser.add_argument("--imgsz",type=int,choices=[640,768,960]);parser.add_argument("--compare-padding",action="store_true")
    parser.add_argument("--backend",choices=["torch","tensorflow"])
    args=parser.parse_args()
    if not args.image.is_file():parser.error(f"Image does not exist: {args.image}")
    pipeline=Pipeline(args.padding,args.imgsz,args.backend)
    try:
        r=pipeline.predict(args.image,paddings=[0,.05,.10,.15,.20] if args.compare_padding else None)
        summary={k:r[k] for k in ["input","plant_gate","detected_leaves","model_load_seconds","timing","annotated_image","json_path"]}
        summary["raw_top3"]=r["raw"]["top3"] if r["raw"] else None
        print(json.dumps(summary,ensure_ascii=False,indent=2))
        for leaf in r["leaves"]:print(json.dumps({"leaf":leaf["id"],"bbox":leaf["bbox"],"yolo_confidence":leaf["yolo_confidence"],"top1":leaf["crop"]["top3"][0],"top3":leaf["crop"]["top3"],"review_flags":leaf["review_flags"]},ensure_ascii=False))
    finally:pipeline.close()
if __name__=="__main__":main()
