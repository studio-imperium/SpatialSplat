from pathlib import Path

from PIL import Image

from training.generate_baselines import has_real_alpha


def test_has_real_alpha_rejects_rgb_and_opaque_rgba(tmp_path: Path) -> None:
    rgb = tmp_path / "rgb.png"
    opaque = tmp_path / "opaque.png"
    Image.new("RGB", (4, 4), "white").save(rgb)
    Image.new("RGBA", (4, 4), (255, 255, 255, 255)).save(opaque)

    assert not has_real_alpha(rgb)
    assert not has_real_alpha(opaque)


def test_has_real_alpha_accepts_transparent_rgba(tmp_path: Path) -> None:
    path = tmp_path / "transparent.png"
    image = Image.new("RGBA", (4, 4), (255, 255, 255, 255))
    image.putpixel((0, 0), (0, 0, 0, 0))
    image.save(path)

    assert has_real_alpha(path)
