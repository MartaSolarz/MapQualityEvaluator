"""Thumbnail generation for the viewer."""

from .config import THUMB_MAX


def make_thumbnail(src_path, thumb_path):
    """Generate a JPEG thumbnail, cached on local disk."""
    from PIL import Image
    try:
        img = Image.open(src_path)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.thumbnail((THUMB_MAX, THUMB_MAX), Image.LANCZOS)
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(thumb_path, format="JPEG", quality=75)
    except Exception:
        import shutil
        shutil.copy2(src_path, thumb_path)
