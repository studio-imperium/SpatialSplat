from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from training.primitive_renderer import render_scene, save_render
from training.scene_schema import OrthographicCamera, Primitive, PrimitiveScene


COLORS = {
    "bark": (112, 72, 42),
    "clay": (174, 88, 58),
    "flower": (190, 72, 138),
    "foliage": (56, 132, 70),
    "grass": (82, 126, 72),
    "ice": (145, 205, 224),
    "metal": (105, 116, 126),
    "rust": (151, 76, 52),
    "sand": (202, 166, 101),
    "snow": (224, 232, 236),
    "soil": (102, 76, 55),
    "stone": (116, 116, 110),
    "water": (58, 137, 174),
    "wood": (143, 96, 57),
    "yellow": (220, 174, 55),
}


@dataclass(frozen=True)
class SceneRecipe:
    scene_id: str
    category: str
    description: str
    ground_material: str | None
    objects: tuple[Primitive, ...]


def _box(name, center, size, color, yaw=0.0) -> Primitive:
    return Primitive(name, "box", center, size, COLORS[color], yaw)


def _sphere(name, center, diameter, color) -> Primitive:
    return Primitive(name, "sphere", center, (diameter,) * 3, COLORS[color])


def _cylinder(name, center, diameter, height, color) -> Primitive:
    return Primitive(name, "cylinder", center, (diameter, height, diameter), COLORS[color])


def _ground(material: str) -> Primitive:
    return _box("ground", (0.0, -0.455, 0.0), (0.98, 0.07, 0.98), material)


def _camera(resolution: int) -> OrthographicCamera:
    return OrthographicCamera(
        position=(1.7, 1.45, 1.7),
        target=(0.0, -0.08, 0.0),
        up=(0.0, 1.0, 0.0),
        ortho_scale=1.45,
        width=resolution,
        height=resolution,
    )


def build_recipes() -> list[SceneRecipe]:
    return [
        SceneRecipe("01_twin_pines", "woodland", "Two distinct pine trees on a grassy forest floor.", "grass", (
            _cylinder("left_trunk", (-0.23, -0.26, 0.02), 0.10, 0.32, "bark"),
            _sphere("left_crown", (-0.23, -0.02, 0.02), 0.30, "foliage"),
            _cylinder("right_trunk", (0.22, -0.28, -0.08), 0.09, 0.28, "bark"),
            _sphere("right_crown", (0.22, -0.07, -0.08), 0.26, "foliage"),
        )),
        SceneRecipe("02_oak_and_boulder", "woodland", "A broad old tree beside one round mossy boulder on grass.", "grass", (
            _cylinder("trunk", (-0.08, -0.24, -0.04), 0.13, 0.36, "bark"),
            _sphere("crown_left", (-0.18, 0.00, -0.02), 0.28, "foliage"),
            _sphere("crown_right", (0.04, 0.02, -0.05), 0.30, "foliage"),
            _sphere("boulder", (0.29, -0.33, 0.20), 0.18, "stone"),
        )),
        SceneRecipe("03_stump_clearing", "woodland", "Three cut tree stumps of different heights in a grassy clearing.", "grass", (
            _cylinder("stump_a", (-0.25, -0.32, 0.16), 0.19, 0.20, "bark"),
            _cylinder("stump_b", (0.02, -0.28, -0.10), 0.22, 0.28, "bark"),
            _cylinder("stump_c", (0.27, -0.34, 0.18), 0.16, 0.16, "bark"),
        )),
        SceneRecipe("04_fallen_log", "woodland", "One long fallen log with two short broken branches on a soil floor.", "soil", (
            _box("log", (0.0, -0.32, 0.0), (0.70, 0.20, 0.20), "bark", 22),
            _box("branch_a", (-0.12, -0.22, 0.10), (0.10, 0.20, 0.10), "bark", 22),
            _box("branch_b", (0.18, -0.24, -0.10), (0.09, 0.16, 0.09), "bark", 22),
        )),
        SceneRecipe("05_forest_gate", "woodland", "A rustic gateway made from two tree trunks and a timber beam, with leafy crowns.", "grass", (
            _cylinder("left_trunk", (-0.25, -0.20, 0.0), 0.13, 0.44, "bark"),
            _cylinder("right_trunk", (0.25, -0.20, 0.0), 0.13, 0.44, "bark"),
            _box("beam", (0.0, 0.06, 0.0), (0.64, 0.12, 0.16), "wood"),
            _sphere("left_leaves", (-0.25, 0.13, 0.0), 0.22, "foliage"),
            _sphere("right_leaves", (0.25, 0.13, 0.0), 0.22, "foliage"),
        )),
        SceneRecipe("06_cactus_pair", "desert", "Two branching cacti of different heights on desert sand.", "sand", (
            _cylinder("left_cactus", (-0.22, -0.20, 0.02), 0.14, 0.44, "foliage"),
            _box("left_arm", (-0.31, -0.19, 0.02), (0.20, 0.09, 0.10), "foliage"),
            _cylinder("right_cactus", (0.24, -0.26, -0.08), 0.12, 0.32, "foliage"),
            _box("right_arm", (0.32, -0.27, -0.08), (0.16, 0.08, 0.09), "foliage"),
        )),
        SceneRecipe("07_sandstone_arch", "desert", "A weathered sandstone arch with a clear opening on sand.", "sand", (
            _box("left", (-0.23, -0.21, 0.0), (0.17, 0.42, 0.22), "clay"),
            _box("right", (0.23, -0.21, 0.0), (0.17, 0.42, 0.22), "clay"),
            _box("lintel", (0.0, 0.04, 0.0), (0.62, 0.12, 0.22), "clay"),
        )),
        SceneRecipe("08_desert_rocks", "desert", "Four separated desert rocks forming a loose curved line on sand.", "sand", (
            _sphere("rock_a", (-0.32, -0.34, 0.18), 0.16, "stone"),
            _box("rock_b", (-0.10, -0.34, 0.05), (0.20, 0.16, 0.18), "clay", 18),
            _sphere("rock_c", (0.14, -0.32, -0.05), 0.20, "stone"),
            _box("rock_d", (0.33, -0.35, -0.20), (0.15, 0.14, 0.17), "clay", -12),
        )),
        SceneRecipe("09_desert_watchtower", "desert", "A cylindrical desert watchtower with a wider square lookout cabin.", "sand", (
            _cylinder("tower", (0.0, -0.18, 0.0), 0.24, 0.48, "clay"),
            _box("lookout", (0.0, 0.10, 0.0), (0.40, 0.16, 0.34), "wood", 8),
        )),
        SceneRecipe("10_palm_oasis", "desert", "A single palm tree beside a small square oasis pool in sand.", "sand", (
            _cylinder("trunk", (-0.18, -0.20, -0.05), 0.11, 0.44, "bark"),
            _sphere("crown_a", (-0.25, 0.08, -0.05), 0.24, "foliage"),
            _sphere("crown_b", (-0.08, 0.08, -0.05), 0.24, "foliage"),
            _box("pool", (0.24, -0.39, 0.16), (0.32, 0.06, 0.28), "water"),
        )),
        SceneRecipe("11_boulder_trio", "rocky", "Three large boulders of descending size on rough stone ground.", "stone", (
            _sphere("large", (-0.24, -0.28, -0.06), 0.28, "stone"),
            _sphere("medium", (0.06, -0.32, 0.08), 0.20, "clay"),
            _sphere("small", (0.29, -0.35, -0.12), 0.14, "stone"),
        )),
        SceneRecipe("12_cairn", "rocky", "A five-level stone cairn with uneven rotated slabs.", "soil", (
            _box("level_1", (0.0, -0.37, 0.0), (0.42, 0.10, 0.32), "stone", 8),
            _box("level_2", (0.01, -0.27, 0.0), (0.34, 0.10, 0.28), "clay", -9),
            _box("level_3", (-0.01, -0.17, 0.0), (0.28, 0.10, 0.23), "stone", 13),
            _box("level_4", (0.01, -0.07, 0.0), (0.20, 0.10, 0.18), "clay", -6),
            _sphere("cap", (0.0, 0.04, 0.0), 0.13, "stone"),
        )),
        SceneRecipe("13_cliff_steps", "rocky", "Four broad rock ledges rising toward the back-right.", "soil", (
            _box("ledge_1", (-0.25, -0.37, 0.24), (0.28, 0.10, 0.28), "stone"),
            _box("ledge_2", (-0.08, -0.31, 0.08), (0.30, 0.22, 0.30), "stone"),
            _box("ledge_3", (0.10, -0.24, -0.10), (0.30, 0.36, 0.30), "stone"),
            _box("ledge_4", (0.27, -0.17, -0.27), (0.27, 0.50, 0.27), "clay"),
        )),
        SceneRecipe("14_balancing_rock", "rocky", "A large round balancing rock resting on a narrow stone pedestal.", "stone", (
            _box("pedestal", (0.0, -0.26, 0.0), (0.22, 0.32, 0.22), "clay", 10),
            _sphere("balancing_rock", (0.0, 0.00, 0.0), 0.28, "stone"),
        )),
        SceneRecipe("15_rock_cave_mouth", "rocky", "A blocky cave entrance with thick side rocks and an uneven roof.", "soil", (
            _box("left_wall", (-0.25, -0.20, 0.0), (0.22, 0.44, 0.32), "stone", -8),
            _box("right_wall", (0.25, -0.22, 0.0), (0.24, 0.40, 0.34), "stone", 9),
            _box("roof", (0.0, 0.05, 0.0), (0.66, 0.16, 0.36), "clay", 4),
        )),
        SceneRecipe("16_snowman", "snow", "A three-ball snowman with two short block arms on snow.", "snow", (
            _sphere("bottom", (0.0, -0.29, 0.0), 0.26, "snow"),
            _sphere("middle", (0.0, -0.09, 0.0), 0.20, "snow"),
            _sphere("head", (0.0, 0.07, 0.0), 0.15, "snow"),
            _box("left_arm", (-0.17, -0.08, 0.0), (0.20, 0.05, 0.05), "bark", -18),
            _box("right_arm", (0.17, -0.08, 0.0), (0.20, 0.05, 0.05), "bark", 18),
        )),
        SceneRecipe("17_snow_pine", "snow", "A snow-covered pine tree beside a low ice boulder.", "snow", (
            _cylinder("trunk", (-0.12, -0.23, -0.04), 0.11, 0.38, "bark"),
            _sphere("crown_low", (-0.12, -0.01, -0.04), 0.32, "foliage"),
            _sphere("crown_high", (-0.12, 0.13, -0.04), 0.22, "snow"),
            _sphere("ice_rock", (0.27, -0.33, 0.18), 0.18, "ice"),
        )),
        SceneRecipe("18_ice_pillars", "snow", "Four translucent ice pillars of varied height on snow.", "snow", (
            _cylinder("pillar_a", (-0.28, -0.24, -0.18), 0.14, 0.36, "ice"),
            _cylinder("pillar_b", (-0.08, -0.16, 0.10), 0.16, 0.52, "ice"),
            _cylinder("pillar_c", (0.16, -0.27, -0.10), 0.13, 0.30, "ice"),
            _cylinder("pillar_d", (0.31, -0.22, 0.20), 0.12, 0.40, "ice"),
        )),
        SceneRecipe("19_snow_cabin", "snow", "A compact snow cabin with a box body, low roof, and small chimney.", "snow", (
            _box("cabin", (0.0, -0.24, 0.0), (0.50, 0.36, 0.42), "wood"),
            _box("roof", (0.0, -0.01, 0.0), (0.62, 0.12, 0.52), "snow", 4),
            _box("chimney", (0.16, 0.10, -0.08), (0.10, 0.22, 0.10), "stone"),
        )),
        SceneRecipe("20_igloo", "snow", "A rounded igloo form with a short rectangular entrance tunnel.", "snow", (
            _sphere("dome", (-0.07, -0.21, -0.05), 0.42, "snow"),
            _box("entrance", (0.20, -0.32, 0.20), (0.24, 0.20, 0.30), "ice", 35),
        )),
        SceneRecipe("21_ruin_columns", "ruins", "Three ancient columns with different heights on a cracked stone floor.", "stone", (
            _cylinder("column_a", (-0.27, -0.18, -0.12), 0.17, 0.48, "stone"),
            _cylinder("column_b", (0.0, -0.25, 0.08), 0.17, 0.34, "clay"),
            _cylinder("column_c", (0.27, -0.12, -0.08), 0.17, 0.60, "stone"),
        )),
        SceneRecipe("22_ruin_altar", "ruins", "A stepped stone altar with a small round relic on top.", "stone", (
            _box("base", (0.0, -0.37, 0.0), (0.62, 0.10, 0.52), "stone"),
            _box("middle", (0.0, -0.27, 0.0), (0.46, 0.10, 0.38), "clay"),
            _box("altar", (0.0, -0.14, 0.0), (0.32, 0.16, 0.28), "stone"),
            _sphere("relic", (0.0, -0.01, 0.0), 0.12, "yellow"),
        )),
        SceneRecipe("23_broken_wall", "ruins", "A broken masonry wall made from five offset blocks.", "soil", (
            _box("block_a", (-0.28, -0.31, 0.0), (0.24, 0.22, 0.20), "stone"),
            _box("block_b", (-0.04, -0.31, 0.0), (0.22, 0.22, 0.20), "clay"),
            _box("block_c", (0.20, -0.31, 0.0), (0.24, 0.22, 0.20), "stone"),
            _box("block_d", (-0.17, -0.09, 0.0), (0.24, 0.22, 0.20), "clay"),
            _box("block_e", (0.08, -0.09, 0.0), (0.22, 0.22, 0.20), "stone"),
        )),
        SceneRecipe("24_obelisk_ring", "ruins", "A tall central obelisk surrounded by four low round markers.", "sand", (
            _box("obelisk", (0.0, -0.13, 0.0), (0.18, 0.58, 0.18), "stone", 8),
            _cylinder("north", (0.0, -0.37, -0.32), 0.16, 0.10, "clay"),
            _cylinder("south", (0.0, -0.37, 0.32), 0.16, 0.10, "clay"),
            _cylinder("west", (-0.32, -0.37, 0.0), 0.16, 0.10, "clay"),
            _cylinder("east", (0.32, -0.37, 0.0), 0.16, 0.10, "clay"),
        )),
        SceneRecipe("25_temple_steps", "ruins", "Three wide temple steps leading to two short pillars.", "stone", (
            _box("step_1", (0.0, -0.38, 0.24), (0.72, 0.08, 0.22), "stone"),
            _box("step_2", (0.0, -0.30, 0.08), (0.62, 0.16, 0.22), "stone"),
            _box("step_3", (0.0, -0.22, -0.08), (0.52, 0.24, 0.22), "clay"),
            _cylinder("left_pillar", (-0.18, -0.04, -0.19), 0.13, 0.52, "stone"),
            _cylinder("right_pillar", (0.18, -0.04, -0.19), 0.13, 0.52, "stone"),
        )),
        SceneRecipe("26_garden_fountain", "garden", "A tiered circular garden fountain on grass.", "grass", (
            _cylinder("basin", (0.0, -0.36, 0.0), 0.56, 0.12, "stone"),
            _cylinder("stem", (0.0, -0.20, 0.0), 0.14, 0.32, "stone"),
            _cylinder("upper_bowl", (0.0, -0.06, 0.0), 0.34, 0.08, "water"),
            _sphere("finial", (0.0, 0.04, 0.0), 0.12, "stone"),
        )),
        SceneRecipe("27_mushroom_grove", "garden", "Five oversized mushrooms of varied height on a mossy floor.", "grass", (
            _cylinder("stem_a", (-0.26, -0.31, 0.13), 0.09, 0.22, "snow"), _sphere("cap_a", (-0.26, -0.17, 0.13), 0.20, "clay"),
            _cylinder("stem_b", (0.0, -0.25, -0.10), 0.10, 0.34, "snow"), _sphere("cap_b", (0.0, -0.05, -0.10), 0.23, "flower"),
            _cylinder("stem_c", (0.27, -0.33, 0.12), 0.08, 0.18, "snow"), _sphere("cap_c", (0.27, -0.21, 0.12), 0.17, "yellow"),
        )),
        SceneRecipe("28_hedge_gate", "garden", "Two dense hedge blocks with a wooden top beam and open passage.", "grass", (
            _box("left_hedge", (-0.25, -0.22, 0.0), (0.22, 0.40, 0.30), "foliage"),
            _box("right_hedge", (0.25, -0.22, 0.0), (0.22, 0.40, 0.30), "foliage"),
            _box("beam", (0.0, 0.04, 0.0), (0.66, 0.12, 0.20), "wood"),
        )),
        SceneRecipe("29_flower_pots", "garden", "Three flower pots with round flowering crowns on soil.", "soil", (
            _cylinder("pot_a", (-0.25, -0.34, 0.10), 0.18, 0.16, "clay"), _sphere("flowers_a", (-0.25, -0.18, 0.10), 0.20, "flower"),
            _cylinder("pot_b", (0.0, -0.31, -0.10), 0.20, 0.22, "clay"), _sphere("flowers_b", (0.0, -0.12, -0.10), 0.23, "yellow"),
            _cylinder("pot_c", (0.27, -0.35, 0.14), 0.15, 0.14, "clay"), _sphere("flowers_c", (0.27, -0.21, 0.14), 0.18, "foliage"),
        )),
        SceneRecipe("30_garden_bench", "garden", "A simple wooden garden bench with seat, back, and four legs.", "grass", (
            _box("seat", (0.0, -0.24, 0.0), (0.68, 0.10, 0.26), "wood"),
            _box("back", (0.0, -0.03, -0.11), (0.68, 0.34, 0.08), "wood"),
            _box("leg_a", (-0.25, -0.34, -0.07), (0.09, 0.20, 0.09), "metal"),
            _box("leg_b", (0.25, -0.34, -0.07), (0.09, 0.20, 0.09), "metal"),
        )),
        SceneRecipe("31_tank_farm", "industrial", "Three industrial storage tanks with different heights on a metal deck.", "metal", (
            _cylinder("tank_a", (-0.25, -0.20, -0.08), 0.24, 0.44, "rust"),
            _cylinder("tank_b", (0.02, -0.25, 0.14), 0.22, 0.34, "metal"),
            _cylinder("tank_c", (0.28, -0.16, -0.12), 0.20, 0.52, "yellow"),
        )),
        SceneRecipe("32_pipe_cluster", "industrial", "A cluster of vertical pipes joined by two horizontal ducts.", "metal", (
            _cylinder("pipe_a", (-0.25, -0.22, 0.0), 0.12, 0.40, "rust"),
            _cylinder("pipe_b", (0.0, -0.16, 0.0), 0.14, 0.52, "metal"),
            _cylinder("pipe_c", (0.25, -0.25, 0.0), 0.11, 0.34, "rust"),
            _box("duct_low", (-0.12, -0.25, 0.0), (0.30, 0.08, 0.10), "metal"),
            _box("duct_high", (0.13, -0.08, 0.0), (0.30, 0.08, 0.10), "yellow"),
        )),
        SceneRecipe("33_conveyor", "industrial", "A long raised conveyor belt supported by four short legs.", "metal", (
            _box("belt", (0.0, -0.18, 0.0), (0.76, 0.14, 0.30), "rust", 12),
            _box("leg_a", (-0.26, -0.33, -0.08), (0.10, 0.24, 0.10), "metal", 12),
            _box("leg_b", (0.26, -0.33, 0.08), (0.10, 0.24, 0.10), "metal", 12),
        )),
        SceneRecipe("34_generator", "industrial", "A boxy generator with one large cylindrical exhaust and a smaller control box.", "metal", (
            _box("main_body", (-0.06, -0.24, 0.0), (0.52, 0.36, 0.42), "yellow", -8),
            _cylinder("exhaust", (0.17, -0.02, -0.10), 0.12, 0.44, "rust"),
            _box("control", (-0.30, -0.25, 0.12), (0.18, 0.30, 0.20), "metal", -8),
        )),
        SceneRecipe("35_crate_yard", "industrial", "Five shipping crates in an uneven two-level arrangement.", "metal", (
            _box("crate_a", (-0.27, -0.31, 0.12), (0.24, 0.22, 0.24), "wood", 5),
            _box("crate_b", (0.0, -0.30, -0.12), (0.28, 0.24, 0.26), "rust", -8),
            _box("crate_c", (0.29, -0.33, 0.14), (0.20, 0.18, 0.22), "metal", 12),
            _box("crate_d", (-0.12, -0.09, 0.05), (0.22, 0.22, 0.22), "yellow", 8),
            _box("crate_e", (0.13, -0.08, -0.07), (0.22, 0.20, 0.22), "wood", -6),
        )),
        SceneRecipe("36_humanoid_robot", "robot", "A standalone friendly humanoid robot with head, torso, two arms, and two legs.", None, (
            _box("left_leg", (-0.10, -0.30, 0.0), (0.12, 0.24, 0.14), "metal"),
            _box("right_leg", (0.10, -0.30, 0.0), (0.12, 0.24, 0.14), "metal"),
            _box("torso", (0.0, -0.08, 0.0), (0.32, 0.28, 0.22), "yellow"),
            _box("head", (0.0, 0.12, 0.0), (0.22, 0.18, 0.20), "metal"),
            _box("left_arm", (-0.23, -0.09, 0.0), (0.14, 0.26, 0.12), "rust", -8),
            _box("right_arm", (0.23, -0.09, 0.0), (0.14, 0.26, 0.12), "rust", 8),
        )),
        SceneRecipe("37_robot_rover", "robot", "A standalone compact rover with a rectangular body, raised sensor, and four round wheels.", None, (
            _box("body", (0.0, -0.24, 0.0), (0.52, 0.24, 0.34), "metal"),
            _box("sensor_mast", (0.0, -0.04, -0.04), (0.10, 0.22, 0.10), "yellow"),
            _sphere("wheel_fl", (-0.25, -0.32, 0.19), 0.16, "rust"),
            _sphere("wheel_fr", (0.25, -0.32, 0.19), 0.16, "rust"),
            _sphere("wheel_bl", (-0.25, -0.32, -0.19), 0.16, "rust"),
            _sphere("wheel_br", (0.25, -0.32, -0.19), 0.16, "rust"),
        )),
        SceneRecipe("38_robot_turret", "robot", "A standalone compact robot turret with round base, square housing, and long barrel.", None, (
            _cylinder("base", (0.0, -0.34, 0.0), 0.40, 0.16, "metal"),
            _box("housing", (0.0, -0.18, 0.0), (0.30, 0.24, 0.28), "yellow"),
            _box("barrel", (0.23, -0.12, 0.0), (0.40, 0.09, 0.10), "rust"),
        )),
        SceneRecipe("39_flying_drone", "robot", "A standalone hovering drone with central round body, four arms, and four motor pods.", None, (
            _sphere("body", (0.0, -0.04, 0.0), 0.24, "metal"),
            _box("arm_x", (0.0, -0.04, 0.0), (0.76, 0.07, 0.08), "yellow", 0),
            _box("arm_z", (0.0, -0.04, 0.0), (0.76, 0.07, 0.08), "rust", 90),
            _sphere("pod_a", (-0.35, -0.04, 0.0), 0.14, "metal"),
            _sphere("pod_b", (0.35, -0.04, 0.0), 0.14, "metal"),
            _sphere("pod_c", (0.0, -0.04, -0.35), 0.14, "metal"),
            _sphere("pod_d", (0.0, -0.04, 0.35), 0.14, "metal"),
        )),
        SceneRecipe("40_cylinder_droid", "robot", "A standalone cylindrical service droid with round head and two block feet.", None, (
            _cylinder("body", (0.0, -0.18, 0.0), 0.30, 0.48, "metal"),
            _sphere("head", (0.0, 0.10, 0.0), 0.26, "yellow"),
            _box("left_foot", (-0.11, -0.38, 0.05), (0.18, 0.10, 0.24), "rust"),
            _box("right_foot", (0.11, -0.38, 0.05), (0.18, 0.10, 0.24), "rust"),
        )),
        SceneRecipe("41_wooden_chair", "object", "A standalone wooden chair with seat, back, and four separated legs.", None, (
            _box("seat", (0.0, -0.20, 0.0), (0.42, 0.10, 0.40), "wood"),
            _box("back", (0.0, 0.02, -0.17), (0.42, 0.36, 0.08), "wood"),
            _box("leg_fl", (-0.15, -0.34, 0.14), (0.08, 0.28, 0.08), "wood"),
            _box("leg_fr", (0.15, -0.34, 0.14), (0.08, 0.28, 0.08), "wood"),
            _box("leg_bl", (-0.15, -0.34, -0.14), (0.08, 0.28, 0.08), "wood"),
            _box("leg_br", (0.15, -0.34, -0.14), (0.08, 0.28, 0.08), "wood"),
        )),
        SceneRecipe("42_floor_lamp", "object", "A standalone floor lamp with round base, thin stem, and large spherical shade.", None, (
            _cylinder("base", (0.0, -0.38, 0.0), 0.34, 0.08, "metal"),
            _cylinder("stem", (0.0, -0.12, 0.0), 0.08, 0.52, "rust"),
            _sphere("shade", (0.0, 0.19, 0.0), 0.28, "yellow"),
        )),
        SceneRecipe("43_toolbox", "object", "A standalone rectangular metal toolbox with a raised block handle.", None, (
            _box("case", (0.0, -0.27, 0.0), (0.64, 0.30, 0.30), "rust", 4),
            _box("handle_left", (-0.16, -0.05, 0.0), (0.08, 0.20, 0.08), "metal", 4),
            _box("handle_right", (0.16, -0.05, 0.0), (0.08, 0.20, 0.08), "metal", 4),
            _box("handle_top", (0.0, 0.03, 0.0), (0.38, 0.08, 0.08), "metal", 4),
        )),
        SceneRecipe("44_trophy", "object", "A standalone trophy with broad round base, narrow stem, cup body, and two handles.", None, (
            _cylinder("base", (0.0, -0.36, 0.0), 0.34, 0.12, "stone"),
            _cylinder("stem", (0.0, -0.21, 0.0), 0.10, 0.22, "yellow"),
            _cylinder("cup", (0.0, -0.02, 0.0), 0.30, 0.22, "yellow"),
            _box("left_handle", (-0.21, -0.02, 0.0), (0.18, 0.08, 0.08), "yellow"),
            _box("right_handle", (0.21, -0.02, 0.0), (0.18, 0.08, 0.08), "yellow"),
        )),
        SceneRecipe("45_toy_house", "object", "A standalone toy house with box body, flat roof slab, chimney, and front step.", None, (
            _box("house", (0.0, -0.21, 0.0), (0.52, 0.42, 0.46), "clay"),
            _box("roof", (0.0, 0.05, 0.0), (0.66, 0.12, 0.58), "rust"),
            _box("chimney", (0.17, 0.16, -0.08), (0.10, 0.24, 0.10), "stone"),
            _box("step", (0.0, -0.37, 0.28), (0.24, 0.10, 0.16), "stone"),
        )),
        SceneRecipe("46_stacked_totem", "abstract", "A standalone asymmetric totem made from alternating boxes, cylinders, and a sphere.", None, (
            _cylinder("base", (0.0, -0.35, 0.0), 0.32, 0.14, "stone"),
            _box("lower", (0.0, -0.22, 0.0), (0.26, 0.16, 0.26), "clay", 18),
            _cylinder("middle", (0.0, -0.07, 0.0), 0.20, 0.18, "metal"),
            _box("upper", (0.0, 0.07, 0.0), (0.22, 0.12, 0.34), "yellow", -14),
            _sphere("cap", (0.0, 0.20, 0.0), 0.16, "flower"),
        )),
        SceneRecipe("47_orbit_sculpture", "abstract", "A standalone central pillar surrounded by four separated orbiting spheres.", None, (
            _cylinder("pillar", (0.0, -0.14, 0.0), 0.16, 0.56, "metal"),
            _sphere("sphere_a", (-0.30, -0.05, 0.0), 0.16, "clay"),
            _sphere("sphere_b", (0.30, -0.05, 0.0), 0.16, "yellow"),
            _sphere("sphere_c", (0.0, 0.12, -0.30), 0.16, "flower"),
            _sphere("sphere_d", (0.0, -0.22, 0.30), 0.16, "ice"),
        )),
        SceneRecipe("48_standalone_bridge", "abstract", "A standalone miniature bridge with two piers, a long deck, and two end ramps.", None, (
            _box("left_pier", (-0.22, -0.29, 0.0), (0.16, 0.26, 0.24), "stone"),
            _box("right_pier", (0.22, -0.29, 0.0), (0.16, 0.26, 0.24), "stone"),
            _box("deck", (0.0, -0.12, 0.0), (0.76, 0.12, 0.28), "rust"),
            _box("left_ramp", (-0.38, -0.22, 0.0), (0.20, 0.10, 0.28), "stone"),
            _box("right_ramp", (0.38, -0.22, 0.0), (0.20, 0.10, 0.28), "stone"),
        )),
        SceneRecipe("49_spiral_blocks", "abstract", "A standalone rising spiral suggested by six separated rotated blocks.", None, (
            _box("block_1", (-0.25, -0.36, 0.20), (0.18, 0.12, 0.18), "clay", 0),
            _box("block_2", (-0.10, -0.29, 0.06), (0.18, 0.18, 0.18), "yellow", 20),
            _box("block_3", (0.08, -0.22, -0.02), (0.18, 0.24, 0.18), "metal", 40),
            _box("block_4", (0.24, -0.15, 0.06), (0.18, 0.30, 0.18), "flower", 60),
            _box("block_5", (0.18, -0.08, 0.24), (0.18, 0.36, 0.18), "ice", 80),
            _box("block_6", (0.0, -0.01, 0.30), (0.18, 0.42, 0.18), "rust", 100),
        )),
        SceneRecipe("50_asymmetric_sculpture", "abstract", "A standalone asymmetrical sculpture with one tall slab, low cylinder, offset sphere, and crossing beam.", None, (
            _box("tall_slab", (-0.20, -0.14, -0.08), (0.18, 0.56, 0.28), "stone", -14),
            _cylinder("low_drum", (0.18, -0.32, 0.16), 0.30, 0.20, "rust"),
            _sphere("offset_sphere", (0.20, -0.02, -0.14), 0.24, "flower"),
            _box("cross_beam", (0.0, 0.07, 0.0), (0.62, 0.10, 0.12), "yellow", 24),
        )),
    ]


def build_scenes(resolution: int) -> list[tuple[PrimitiveScene, SceneRecipe]]:
    camera = _camera(resolution)
    result = []
    for recipe in build_recipes():
        primitives = recipe.objects
        if recipe.ground_material is not None:
            primitives = (_ground(recipe.ground_material), *primitives)
        result.append(
            (
                PrimitiveScene(
                    recipe.scene_id, recipe.description, camera, tuple(primitives)
                ),
                recipe,
            )
        )
    return result


def chroma_key(recipe: SceneRecipe) -> str:
    uses_green = recipe.ground_material == "grass" or any(
        primitive.color == COLORS["foliage"] for primitive in recipe.objects
    )
    return "#ff00ff" if uses_green else "#00ff00"


def diverse_generation_prompt(scene: PrimitiveScene, recipe: SceneRecipe) -> str:
    chroma = chroma_key(recipe)
    floor_rule = (
        "The ground slab is part of the subject and must remain fully opaque and visible."
        if recipe.ground_material is not None
        else "This is a standalone object with no floor, platform, terrain, or ground plane."
    )
    return (
        "Transform the attached primitive proxy into a richly rendered, believable miniature scene while preserving "
        "the exact fixed isometric camera, object count, broad silhouettes, relative positions, sizes, and occlusion order. "
        "Decorative texture and small details are encouraged, but do not move, remove, merge, or add any large object. "
        f"Scene description: {scene.description} "
        "Interpret the simple colors as the natural materials named in the scene description, not as a required palette. "
        + floor_rule
        + f" Render everything outside the subject on a perfectly flat solid {chroma} chroma-key background. "
        "The chroma background must have no gradient, texture, horizon, cast shadow, or reflection. "
        "Do not use the chroma-key color on the subject."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 50 diverse primitive contracts and image prompts."
    )
    parser.add_argument("--output", type=Path, default=Path("poc_data/diverse_train"))
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    for scene, recipe in build_scenes(args.resolution):
        scene_dir = args.output / scene.scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        scene.write_json(scene_dir / "scene.json")
        prompt = diverse_generation_prompt(scene, recipe)
        (scene_dir / "generation_prompt.txt").write_text(
            prompt + "\n", encoding="utf-8"
        )
        metadata = {
            "scene_id": recipe.scene_id,
            "category": recipe.category,
            "description": recipe.description,
            "ground_material": recipe.ground_material,
            "has_floor": recipe.ground_material is not None,
            "chroma_key": chroma_key(recipe),
            "requires_real_alpha": True,
            "training_split": "diverse_train",
            "object_names": [primitive.name for primitive in recipe.objects],
        }
        (scene_dir / "condition_spec.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        save_render(render_scene(scene), scene_dir)
        print(scene_dir)


if __name__ == "__main__":
    main()
