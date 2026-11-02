import argparse,csv,json,sys,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.pipeline import Pipeline
from src.config import HOME

def main():
    p=argparse.ArgumentParser();p.add_argument("folder",type=Path);p.add_argument("--limit",type=int);p.add_argument("--compare-padding",action="store_true");p.add_argument("--backend",choices=["torch","tensorflow"]);a=p.parse_args()
    paths=sorted(x for x in a.folder.rglob("*") if x.suffix.lower() in {".jpg",".jpeg",".png",".webp",".bmp"})
    if a.limit is not None:paths=paths[:a.limit]
    if not paths:raise SystemExit("No images found")
    pipe=Pipeline(backend=a.backend);rows=[];errors=[]
    try:
        for path in paths:
            try:
                r=pipe.predict(path,paddings=[0,.05,.1,.15,.2] if a.compare_padding else None)
                for leaf in r["leaves"] or [None]:
                    rows.append({"input":str(path),"gate":r["plant_gate"]["label"],"detections":r["detected_leaves"],"leaf":leaf["id"] if leaf else "","bbox":json.dumps(leaf["bbox"]) if leaf else "","yolo_confidence":leaf["yolo_confidence"] if leaf else "","crop_top1":leaf["crop"]["top3"][0]["class"] if leaf else "","crop_confidence":leaf["crop"]["top3"][0]["confidence"] if leaf else "","raw_top1":r["raw"]["top3"][0]["class"] if r["raw"] else "","fusion_top1":leaf["experimental_fusion"]["top3"][0]["class"] if leaf else "","seconds":r["timing"]["pipeline_seconds"],"flags":json.dumps(leaf["review_flags"] if leaf else r["review_flags"]),"json":r["json_path"]})
                print(f"{path.name}: {r['plant_gate']['label']}, {r['detected_leaves']} leaves",flush=True)
            except Exception as exc:errors.append({"input":str(path),"error":str(exc)});print(f"FAILED {path}: {exc}",file=sys.stderr)
    finally:pipe.close()
    token=uuid.uuid4().hex[:12];dest=HOME/"outputs/metrics"/f"batch_{token}.csv"
    if rows:
        with dest.open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (HOME/"outputs/failed"/f"errors_{token}.json").write_text(json.dumps(errors,indent=2),encoding="utf-8")
    print(f"Images: {len(paths)}, failures: {len(errors)}, CSV: {dest}")
    if errors:raise SystemExit(1)
if __name__=="__main__":main()
