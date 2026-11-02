import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from src.classifier import preprocess
from src.pipeline import crop_box,fusion

class CoreTests(unittest.TestCase):
    def test_distinct_preprocessing(self):
        im=Image.new("RGB",(10,10),(255,128,0))
        gate=preprocess(im,normalize=False);disease=preprocess(im)
        self.assertEqual(gate.shape,(224,224,3));self.assertEqual(gate.dtype,np.float32)
        np.testing.assert_allclose(gate[0,0],[255,128,0]);np.testing.assert_allclose(disease[0,0],[1,128/255,0])
    def test_padding_clips_and_rejects_invalid_box(self):
        self.assertEqual(crop_box([0,0,100,100],(100,100),.2),[0,0,100,100])
        self.assertEqual(crop_box([20,20,60,60],(100,100),.1),[16,16,64,64])
        with self.assertRaises(ValueError):crop_box([60,20,20,60],(100,100),0)
    def test_fusion_confidence_boundary(self):
        raw=np.array([.7,.3]);crop=np.array([.1,.9])
        np.testing.assert_allclose(fusion(raw,crop,.7,.4),raw)
        np.testing.assert_allclose(fusion(np.array([.6,.4]),crop,.7,.4),[.3,.7])
    def test_gate_thresholds_from_config(self):
        from src.config import load_config
        c=load_config();self.assertEqual(len(c["classes"]),38)
        self.assertEqual(c["gate"]["plant_threshold"],.8)
        self.assertEqual(c["gate"]["uncertain_lower_threshold"],.4)
class MatchingTests(unittest.TestCase):
    def test_duplicate_prediction_does_not_double_count(self):
        import importlib.util
        script=Path(__file__).resolve().parents[1]/"scripts/benchmark_detector.py"
        spec=importlib.util.spec_from_file_location("benchmark",script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        boxes=[{"bbox":[0,0,10,10],"yolo_confidence":.9},{"bbox":[0,0,10,10],"yolo_confidence":.8}]
        stats=module.match(boxes,[[0,0,10,10,True]])
        self.assertEqual(stats,{"tp":1,"fp":1,"fn":0,"small_tp":1,"small_gt":1})
class PipelineControlTests(unittest.TestCase):
    def make_pipeline(self,label,boxes):
        from src.pipeline import Pipeline
        from src.config import load_config,ROOT
        from unittest.mock import Mock
        pipe=Pipeline.__new__(Pipeline);pipe.cfg=load_config();pipe.load_seconds=0;pipe.run="unit"
        pipe.classifier=Mock();pipe.classifier.plant.return_value={"label":label,"plant_probability":.1 if label=="not_plant" else .6}
        pipe.detector=Mock();pipe.detector.metadata={};pipe.detector.predict.return_value={"boxes":boxes,"seconds":.01,"speed_ms":{}}
        def probabilities(images):
            p=np.zeros((len(images),38),dtype=np.float32)
            for i in range(len(images)):p[i,i]=1
            return p
        pipe.classifier.predict.side_effect=probabilities
        return pipe,ROOT/"test_leaf_image_for_gradio.jpg"
    def test_gate_rejection_skips_detector_and_disease(self):
        pipe,path=self.make_pipeline("not_plant",[]);result=pipe.predict(path,save=False)
        self.assertEqual(result["status"],"gate_rejected");pipe.detector.predict.assert_not_called();pipe.classifier.predict.assert_not_called()
    def test_uncertain_gate_keeps_independent_leaf_predictions(self):
        boxes=[{"bbox":[10,10,50,50],"yolo_confidence":.9,"class_id":0},{"bbox":[80,80,150,150],"yolo_confidence":.8,"class_id":0}]
        pipe,path=self.make_pipeline("uncertain",boxes);result=pipe.predict(path,save=False)
        self.assertEqual(result["detected_leaves"],2)
        self.assertEqual([l["crop"]["top3"][0]["index"] for l in result["leaves"]],[1,2])
        self.assertEqual(result["raw"]["top3"][0]["index"],0)
        self.assertIn("uncertain_gate",result["review_flags"])
if __name__=="__main__":unittest.main()
