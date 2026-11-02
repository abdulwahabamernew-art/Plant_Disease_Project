"""Verify the opt-in review never runs or fabricates disease classification."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import app

class Gate:
    def plant(self,image):return {'label':'not_plant','plant_probability':.0002}
    def predict(self,*args):raise AssertionError('Diagnostic must not classify disease')
class Detector:
    def predict(self,path):return {'boxes':[{'bbox':[8,8,60,60],'yolo_confidence':.7}]}
class FakePipeline:
    classifier=Gate()
    detector=Detector()

class DiagnosticTests(unittest.TestCase):
    def test_rejected_gate_is_preserved_and_disease_is_not_run(self):
        base=app.UI/'verification/test_inputs';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as directory:
            root=Path(directory);source=root/'input.png';Image.new('RGB',(80,80),'green').save(source)
            with patch.object(app,'UI',root),patch.object(app,'_pipeline',FakePipeline()):
                _,summary,files=app.detector_only(source,progress=lambda *a,**k:None)
                record=json.loads(Path(files[1]).read_text(encoding='utf-8'))
                self.assertEqual(record['plant_gate']['label'],'not_plant')
                self.assertFalse(record['disease_classification_performed'])
                self.assertIn('لا تتضمن تشخيص مرض',summary)
                self.assertEqual(len(record['detection']['boxes']),1)
    def test_cleared_upload_disables_analysis_and_clears_diagnostics(self):
        values=app.preview(None)
        self.assertFalse(values[7]['interactive'])
        self.assertEqual(values[-3:],(None,'',None))

if __name__=='__main__':unittest.main()
