from __future__ import annotations

from pathlib import Path

from PIL import Image


def display_video(video_path: Path, *, width: int = 960) -> object:
    from IPython.display import Video, display

    video = Video(str(video_path), embed=True, width=width)
    display(video)
    return video


def display_image(image: Path | Image.Image) -> object:
    from IPython.display import display

    if isinstance(image, Image.Image):
        display(image)
        return image

    loaded = Image.open(image)
    display(loaded)
    return loaded
