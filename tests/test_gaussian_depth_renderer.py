import numpy as np

from training.gaussian_depth_renderer import GaussianCloud, render_gaussian_depth


def _cloud(means: list[list[float]], opacities: list[float]) -> GaussianCloud:
    count = len(means)
    return GaussianCloud(
        means=np.asarray(means, dtype=np.float32),
        scales=np.full((count, 3), 0.08, dtype=np.float32),
        rotations=np.tile(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (count, 1)
        ),
        opacities=np.asarray(opacities, dtype=np.float32),
    )


def test_single_gaussian_renders_depth_and_alpha() -> None:
    rendered = render_gaussian_depth(_cloud([[0.0, 0.0, 0.0]], [0.99]), 64, 64)

    assert rendered.depth.shape == (64, 64)
    assert rendered.alpha.shape == (64, 64)
    assert rendered.alpha.max() > 0.9
    assert rendered.depth[32, 32] > 1.0


def test_front_gaussian_dominates_depth() -> None:
    rendered = render_gaussian_depth(
        _cloud([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]], [0.999, 0.999]),
        64,
        64,
    )

    center_depth = rendered.depth[32, 32]
    assert 1.0 < center_depth < 1.8


def test_empty_cloud_returns_empty_images() -> None:
    rendered = render_gaussian_depth(_cloud([], []), 16, 12)

    assert not rendered.alpha.any()
    assert not rendered.depth.any()
