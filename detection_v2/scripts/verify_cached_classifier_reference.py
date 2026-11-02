"""Compare frozen classifier against saved notebook-20 TensorFlow crop probabilities."""
import json,sys,hashlib,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from src.config import HOME,ROOT,load_config
from src.classifier import Classifier

def main():
    crops_path=ROOT/"results/detector_assisted_audit_v2_safe/stage_a_detector/chunks/chunk_0000_crops.npz"
    ref_path=ROOT/"results/raw_crop_fusion_audit_v1/probability_worker/chunks/chunk_0000_probs.npz"
    with np.load(crops_path,allow_pickle=False) as z:
        crops=z["crops"];owners=z["source_global_index"];ranks=z["rank"];conf=z["det_conf"]
    with np.load(ref_path,allow_pickle=False) as z:
        ids=z["audit_index"];refs_top=z["top_probs"];refs_multi=z["multi_probs"]
    cfg=load_config();classifier=Classifier(cfg)
    probs=classifier.predict([Image.fromarray(x) for x in crops])
    top=[];multi=[];ref_top=[];ref_multi=[];checked=[]
    for row,idx in enumerate(ids):
        selected=np.where(owners==idx)[0]
        if not len(selected):continue
        selected=selected[np.argsort(ranks[selected])]
        top.append(probs[selected[0]]);ref_top.append(refs_top[row]);checked.append(int(idx))
        w=conf[selected].astype(np.float32);w/=max(float(w.sum()),1e-12)
        m=np.sum(probs[selected]*w[:,None],axis=0);m/=max(float(m.sum()),1e-12)
        multi.append(m);ref_multi.append(refs_multi[row])
    if not top:raise RuntimeError("No reference crops")
    def compare(a,b):
        a=np.stack(a);b=np.stack(b)
        return {"max_abs_difference":float(np.max(np.abs(a-b))),"mean_abs_difference":float(np.mean(np.abs(a-b))),"top1_agreement":float(np.mean(a.argmax(axis=1)==b.argmax(axis=1))),"allclose_atol_1e4_rtol_1e3":bool(np.allclose(a,b,atol=1e-4,rtol=1e-3))}
    cpu_path=ROOT/"results/detector_assisted_audit_v2_safe/stage_b_classifier/chunks/chunk_0000_predictions.csv"
    cpu_rows={int(r["audit_index"]):r for r in csv.DictReader(cpu_path.open(encoding="utf-8-sig"))}
    current=np.stack(top);cpu_conf=np.array([float(cpu_rows[i]["top1_confidence"]) for i in checked])
    cpu_labels=np.array([int(cpu_rows[i]["top1_pred_label"]) for i in checked])
    cpu_top3=[json.loads(cpu_rows[i]["top1_top3"]) for i in checked]
    current_top3=np.argsort(current,axis=1)[:,::-1][:,:3].tolist()
    cpu={"reference":str(cpu_path),"max_abs_top1_confidence_difference":float(np.max(np.abs(current.max(axis=1)-cpu_conf))),"top1_agreement":float(np.mean(current.argmax(axis=1)==cpu_labels)),"top3_order_agreement":float(np.mean([a==b for a,b in zip(current_top3,cpu_top3)])),"confidence_allclose_atol_1e4_rtol_1e3":bool(np.allclose(current.max(axis=1),cpu_conf,atol=1e-4,rtol=1e-3))}
    result={"notebook19_cpu_reference":cpu,"current_top_probabilities":current.tolist(),"backend":cfg["classifier_backend"],"crop_count":len(crops),"reference_rows":len(top),"audit_indices":checked,"top_crop":compare(top,ref_top),"weighted_multi_crop":compare(multi,ref_multi),"classifier_sha256":hashlib.sha256(Path(cfg["models"]["disease"]).read_bytes()).hexdigest(),"reference":str(ref_path),"crop_source":str(crops_path),"limitations":"Stored TensorFlow probabilities, not a fresh TensorFlow execution. Covers cached disease-classifier crops only; does not verify plant gate parity or full-frame preprocessing. No accuracy score inferred."}
    dest=HOME/"outputs/metrics/cached_classifier_reference_verification.json";dest.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="current_top_probabilities"},indent=2))
    if not result["top_crop"]["allclose_atol_1e4_rtol_1e3"] or not result["weighted_multi_crop"]["allclose_atol_1e4_rtol_1e3"]:raise SystemExit(1)
if __name__=="__main__":main()
