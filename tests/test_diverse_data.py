from training.generate_diverse_data import build_recipes, build_scenes


def test_diverse_dataset_has_fifty_unique_scenes() -> None:
    recipes = build_recipes()

    assert len(recipes) == 50
    assert len({recipe.scene_id for recipe in recipes}) == 50
    assert len({recipe.category for recipe in recipes}) >= 10


def test_diverse_dataset_mixes_grounded_and_standalone_scenes() -> None:
    recipes = build_recipes()
    grounded = [recipe for recipe in recipes if recipe.ground_material is not None]
    standalone = [recipe for recipe in recipes if recipe.ground_material is None]

    assert len(grounded) == 35
    assert len(standalone) == 15
    assert all(scene.camera.width == 64 for scene, _ in build_scenes(64))
