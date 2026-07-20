import numpy as np
from PIL import Image

from training.add_condition_alpha import border_connected_alpha


def test_border_connected_alpha_keeps_dark_platform_and_light_object() -> None:
    array = np.full((32, 32, 3), 230, dtype=np.uint8)
    array[18:28, 4:28] = 70
    array[8:20, 12:20] = 205
    array[8:20, 12] = 80
    array[8:20, 19] = 80
    array[8, 12:20] = 80
    array[19, 12:20] = 80

    result = border_connected_alpha(
        Image.fromarray(array), feather_radius=0
    )
    alpha = np.asarray(result.getchannel("A"))

    assert alpha[0, 0] == 0
    assert alpha[24, 16] == 255
    assert alpha[12, 16] == 255
