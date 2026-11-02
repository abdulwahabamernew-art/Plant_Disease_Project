import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.pipeline import checked_output_root
from src.config import HOME,ROOT
class OutputScopeTests(unittest.TestCase):
    def test_new_round_is_inside_allowed_folder(self):
        path=HOME/"outputs/natural_review_20261006/results"
        self.assertEqual(checked_output_root(path),path.resolve())
    def test_original_project_folders_are_not_output_targets(self):
        with self.assertRaises(ValueError):checked_output_root(ROOT/"results")
if __name__=="__main__":unittest.main()
