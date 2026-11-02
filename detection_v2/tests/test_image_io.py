import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unittest.mock import patch
import numpy as np
from PIL import Image
from src.image_io import load_rgb
class OrientationTests(unittest.TestCase):
    def test_rotates_pixels_without_serializing_bad_exif(self):
        data=np.arange(18,dtype=np.uint8).reshape(2,3,3)
        image=Image.fromarray(data)
        class FragileExif(dict):
            def tobytes(self):raise OverflowError("malformed metadata must not be serialized")
        image.getexif=lambda:FragileExif({274:6})
        with patch("src.image_io.Image.open",return_value=image):
            result=load_rgb("unused.jpg")
        self.assertEqual(result.size,(2,3))
        np.testing.assert_array_equal(np.asarray(result),np.rot90(data,k=3))
        self.assertNotIn("exif",result.info)
    def test_all_exif_orientation_transforms(self):
        for orientation,transform in [(2,Image.Transpose.FLIP_LEFT_RIGHT),(3,Image.Transpose.ROTATE_180),(4,Image.Transpose.FLIP_TOP_BOTTOM),(5,Image.Transpose.TRANSPOSE),(6,Image.Transpose.ROTATE_270),(7,Image.Transpose.TRANSVERSE),(8,Image.Transpose.ROTATE_90)]:
            with self.subTest(orientation=orientation):
                data=np.arange(18,dtype=np.uint8).reshape(2,3,3);image=Image.fromarray(data)
                expected=np.asarray(image.transpose(transform));image.getexif=lambda:{274:orientation}
                with patch("src.image_io.Image.open",return_value=image):result=load_rgb("unused.jpg")
                np.testing.assert_array_equal(np.asarray(result),expected)
if __name__=="__main__":unittest.main()
