"""Decode and orient RGB pixels without rewriting fragile camera EXIF metadata."""
from pathlib import Path
from PIL import Image
_ORIENTATION={2:Image.Transpose.FLIP_LEFT_RIGHT,3:Image.Transpose.ROTATE_180,4:Image.Transpose.FLIP_TOP_BOTTOM,5:Image.Transpose.TRANSPOSE,6:Image.Transpose.ROTATE_270,7:Image.Transpose.TRANSVERSE,8:Image.Transpose.ROTATE_90}
def load_rgb(path):
    with Image.open(Path(path)) as source:
        orientation=source.getexif().get(274,1)
        image=source.convert("RGB")
        if orientation in _ORIENTATION:image=image.transpose(_ORIENTATION[orientation])
        # Pixel orientation is already normalized; do not propagate stale orientation
        # or attempt to serialize unrelated malformed GPS/maker metadata.
        image.info.clear()
        return image
