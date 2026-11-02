import unittest
from src.ui_presenter import present,class_label

class PresentationTests(unittest.TestCase):
    def test_rejected_gate_does_not_invent_diagnoses(self):
        result={'plant_gate':{'label':'not_plant'},'detected_leaves':0,'leaves':[],'timing':{'pipeline_seconds':.2}}
        summary,rows,gallery,top=present(result)
        self.assertIn('توقف التحليل',summary)
        self.assertEqual((rows,gallery,top),([],[],[]))
    def test_uncertain_gate_marks_even_confident_leaf_for_review(self):
        p={'class':'Apple___healthy','confidence':.99}
        leaf={'id':1,'crop':{'top3':[p]},'padding_variants':[{'saved_crop':'crop.jpg'}],'review_flags':[]}
        result={'plant_gate':{'label':'uncertain'},'detected_leaves':1,'leaves':[leaf],'timing':{'pipeline_seconds':1}}
        _,rows,_,_=present(result)
        self.assertIn('بوابة النبات غير متأكدة',rows[0][-1])
        self.assertEqual(class_label(p['class']),('تفاح','سليم بحسب التوقع'))

if __name__=='__main__':unittest.main()
