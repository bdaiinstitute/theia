# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from collections import OrderedDict
from typing import Optional

"""
This ALL_OXE_DATASETS below records metadata of all subsets of OXE dataset.
The datasets are in alphabetical order.

versions (list[str]): available and usable versions, sorted from older to newer.
                      Usually use the last one.
episodes (int): total episodes in the dataset.
steps    (int): total steps in the dataset.
visual_observation_keys (list[str]): keys to specify image observations.
"""
ALL_OXE_DATASETS: OrderedDict = OrderedDict(
    {
        "agent_aware_affordances": {
            "versions": ["1.0.0"],
            "episodes": 118,
            "steps": 151628,
            "visual_observation_keys": ["image"],
        },
        "asu_table_top_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 110,
            "steps": 26113,
            "visual_observation_keys": ["image"],
        },
        "austin_buds_dataset_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 50,
            "steps": 34112,
            "visual_observation_keys": ["image", "wrist_image"],
        },
        "austin_sailor_dataset_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 240,
            "steps": 353094,
            "visual_observation_keys": ["image", "wrist_image"],
        },
        "austin_sirius_dataset_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 559,
            "steps": 279939,
            "visual_observation_keys": ["image", "wrist_image"],
        },
        "bc_z": {
            "versions": [
                "0.1.0",  # "1.0.0", "old1.0.1", and "1.0.1" are not usable
            ],
            "episodes": 39350,
            "steps": 5471693,
            "visual_observation_keys": ["image"],
        },
        "berkeley_autolab_ur5": {
            "versions": ["0.1.0"],
            "episodes": 896,
            "steps": 87783,
            "visual_observation_keys": ["image", "hand_image"],
        },
        "berkeley_cable_routing": {
            "versions": ["0.1.0"],
            "episodes": 1482,
            "steps": 38240,
            "visual_observation_keys": ["image", "top_image", "wrist225_image", "wrist45_image"],
        },
        "berkeley_fanuc_manipulation": {
            "versions": ["0.1.0"],
            "episodes": 415,
            "steps": 62613,
            "visual_observation_keys": ["image", "wrist_image"],
        },
        "berkeley_gnm_cory_hall": {
            "versions": ["0.1.0"],
            "episodes": 7331,
            "steps": 156012,
            "visual_observation_keys": ["image"],
        },
        "berkeley_gnm_recon": {
            "versions": ["0.1.0"],
            "episodes": 11834,
            "steps": 610907,
            "visual_observation_keys": ["image"],
        },
        "berkeley_gnm_sac_son": {
            "versions": ["0.1.0"],
            "episodes": 2955,
            "steps": 241059,
            "visual_observation_keys": ["image"],
        },
        "berkeley_mvp_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 480,
            "steps": 45308,
            "visual_observation_keys": ["hand_image"],
        },
        "berkeley_rpt_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 908,
            "steps": 392578,
            "visual_observation_keys": ["hand_image"],
        },
        "bridge": {"versions": ["0.1.0"], "episodes": 25460, "steps": 864292, "visual_observation_keys": ["image"]},
        "cmu_franka_exploration_dataset_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 199,
            "steps": 1990,
            "visual_observation_keys": ["image"],
        },
        "cmu_play_fusion": {
            "versions": ["0.1.0"],
            "episodes": 576,
            "steps": 235922,
            "visual_observation_keys": ["image"],
        },
        "cmu_playing_with_food": {  # this dataset seems to be corrupted
            "versions": ["1.0.0"],
            "episodes": 4200,
            "steps": 83240,
            "visual_observation_keys": ["image"],
        },
        "cmu_stretch": {"versions": ["0.1.0"], "episodes": 135, "steps": 25016, "visual_observation_keys": ["image"]},
        "columbia_cairlab_pusht_real": {
            "versions": ["0.1.0"],
            "episodes": 122,
            "steps": 24924,
            "visual_observation_keys": ["image", "wrist_image"],
        },
        "dlr_edan_shared_control_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 104,
            "steps": 8928,
            "visual_observation_keys": ["image"],
        },
        "dlr_sara_grid_clamp_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 107,
            "steps": 7622,
            "visual_observation_keys": ["image"],
        },
        "dlr_sara_pour_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 100,
            "steps": 12971,
            "visual_observation_keys": ["image"],
        },
        "eth_agent_affordances": {
            "versions": ["0.1.0"],
            "episodes": 118,
            "steps": 151628,
            "visual_observation_keys": ["image"],
        },
        "fanuc_manipulation_v2": {
            "versions": ["1.0.0"],
            "episodes": 415,
            "steps": 62613,
            "visual_observation_keys": ["image", "wrist_image"],
        },
        "fractal20220817_data": {
            "versions": ["0.1.0"],
            "episodes": 87212,
            "steps": 3786400,
            "visual_observation_keys": ["image"],
        },
        "furniture_bench_dataset_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 5100,
            "steps": 3948057,
            "visual_observation_keys": ["image", "wrist_image"],
        },
        "iamlab_cmu_pickup_insert_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 631,
            "steps": 146241,
            "visual_observation_keys": ["image", "wrist_image"],
        },
        "imperial_wrist_dataset": {
            "versions": ["1.0.0"],
            "episodes": 170,
            "steps": 7148,
            "visual_observation_keys": ["image", "wrist_image"],
        },
        "imperialcollege_sawyer_wrist_cam": {
            "versions": ["0.1.0"],
            "episodes": 170,
            "steps": 7148,
            "visual_observation_keys": ["image", "wrist_image"],
        },
        "jaco_play": {
            "versions": ["0.1.0"],
            "episodes": 976,
            "steps": 70127,
            "visual_observation_keys": ["image", "image_wrist"],
        },
        "kaist_nonprehensile_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 201,
            "steps": 32429,
            "visual_observation_keys": ["image"],
        },
        "kuka": {"versions": ["0.1.0"], "episodes": 580392, "steps": 8583978, "visual_observation_keys": ["image"]},
        "language_table": {
            "versions": ["0.0.1", "0.1.0"],
            "episodes": 442226,
            "steps": 7045476,
            "visual_observation_keys": ["rgb"],
        },
        "language_table_blocktoabsolute_oracle_sim": {
            "versions": ["0.0.1"],
            "episodes": 200000,
            "steps": 15866385,
            "visual_observation_keys": ["rgb"],
        },
        "language_table_blocktoblock_4block_sim": {
            "versions": ["0.0.1"],
            "episodes": 8298,
            "steps": 326768,
            "visual_observation_keys": ["rgb"],
        },
        "language_table_blocktoblock_oracle_sim": {
            "versions": ["0.0.1"],
            "episodes": 200000,
            "steps": 12970620,
            "visual_observation_keys": ["rgb"],
        },
        "language_table_blocktoblock_sim": {
            "versions": ["0.0.1"],
            "episodes": 8000,
            "steps": 351688,
            "visual_observation_keys": ["rgb"],
        },
        "language_table_blocktoblockrelative_oracle_sim": {
            "versions": ["0.0.1"],
            "episodes": 200000,
            "steps": 13016749,
            "visual_observation_keys": ["rgb"],
        },
        "language_table_blocktorelative_oracle_sim": {
            "versions": ["0.0.1"],
            "episodes": 200000,
            "steps": 8655815,
            "visual_observation_keys": ["rgb"],
        },
        "language_table_separate_oracle_sim": {
            "versions": ["0.0.1"],
            "episodes": 200000,
            "steps": 3196661,
            "visual_observation_keys": ["rgb"],
        },
        "language_table_sim": {
            "versions": ["0.0.1"],
            "episodes": 181020,
            "steps": 4665423,
            "visual_observation_keys": ["rgb"],
        },
        "maniskill_dataset_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 30213,
            "steps": 4537402,
            "visual_observation_keys": ["image", "wrist_image"],
        },
        "mutex_dataset": {
            "versions": ["1.0.0"],
            "episodes": 1500,
            "steps": 361883,
            "visual_observation_keys": ["image", "wrist_image"],
        },
        "nyu_door_opening_surprising_effectiveness": {
            "versions": ["0.1.0"],
            "episodes": 435,
            "steps": 18196,
            "visual_observation_keys": ["image"],
        },
        "nyu_franka_play_dataset_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 365,
            "steps": 34448,
            "visual_observation_keys": ["image", "image_additional_view"],
        },
        "nyu_rot_dataset_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 14,
            "steps": 440,
            "visual_observation_keys": ["image"],
        },
        "qut_dexterous_manpulation": {
            "versions": ["0.1.0"],
            "episodes": 200,
            "steps": 176278,
            "visual_observation_keys": ["image", "wrist_image"],
        },
        "robo_net": {
            "versions": ["0.1.0", "1.0.0"],
            "episodes": 82775,
            "steps": 2483250,
            "visual_observation_keys": ["image", "image1", "image2"],
        },
        "robot_vqa": {
            "versions": ["0.1.0"],
            "episodes": 3331523,
            "steps": 3331523,
            "visual_observation_keys": ["images"],
        },
        "roboturk": {
            "versions": ["0.1.0"],
            "episodes": 1796,
            "steps": 168423,
            "visual_observation_keys": ["front_rgb"],
        },
        "stanford_hydra_dataset_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 570,
            "steps": 358234,
            "visual_observation_keys": ["image", "wrist_image"],
        },
        "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 3000,
            "steps": 149985,
            "visual_observation_keys": ["image"],
        },
        "stanford_mask_vit_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 9109,
            "steps": 282379,
            "visual_observation_keys": ["image"],
        },
        "stanford_robocook_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 2460,
            "steps": 112980,
            "visual_observation_keys": ["image_1", "image_2", "image_3", "image_4"],
        },
        "taco_play": {
            "versions": ["0.1.0"],
            "episodes": 3242,
            "steps": 213972,
            "visual_observation_keys": ["rgb_static", "rgb_gripper"],
        },
        "tokyo_u_lsmo_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 50,
            "steps": 11925,
            "visual_observation_keys": ["image"],
        },
        "toto": {"versions": ["0.1.0"], "episodes": 902, "steps": 294139, "visual_observation_keys": ["image"]},
        "ucsd_kitchen_dataset_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 150,
            "steps": 3970,
            "visual_observation_keys": ["image"],
        },
        "ucsd_pick_and_place_dataset_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 1355,
            "steps": 67750,
            "visual_observation_keys": ["image"],
        },
        "uiuc_d3field": {  # this dataset seems to be corrupted
            "versions": ["0.1.0", "1.1.2"],
            "episodes": 196,
            "steps": 13384,
            "visual_observation_keys": ["image_1", "image_2", "image_3", "image_4"],
        },
        "usc_cloth_sim_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 800,
            "steps": 80000,
            "visual_observation_keys": ["image"],
        },
        "utaustin_mutex": {
            "versions": ["0.1.0"],
            "episodes": 1500,
            "steps": 361883,
            "visual_observation_keys": ["image", "wrist_image"],
        },
        "utokyo_pr2_opening_fridge_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 64,
            "steps": 9140,
            "visual_observation_keys": ["image"],
        },
        "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 192,
            "steps": 26346,
            "visual_observation_keys": ["image"],
        },
        "utokyo_saytap_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 20,
            "steps": 22937,
            "visual_observation_keys": ["image", "wrist_image"],
        },
        "utokyo_xarm_bimanual_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 64,
            "steps": 1388,
            "visual_observation_keys": ["image"],
        },
        "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {
            "versions": ["0.1.0"],
            "episodes": 92,
            "steps": 6789,
            "visual_observation_keys": ["image", "hand_image", "image2"],
        },
        "viola": {
            "versions": ["0.1.0"],
            "episodes": 135,
            "steps": 68913,
            "visual_observation_keys": ["agentview_rgb", "eye_in_hand_rgb"],
        },
    }
)


def oxe_dsname2path(dataset_name: str, version: Optional[str] = None) -> str:
    """From dataset name to remote google clound path to the dataset.

    Args:
        dataset_name (str): dataset name.
        version (Optional[str]): version string.

    Returns:
        str: google clound path
    """
    if version is None:
        version = ALL_OXE_DATASETS[dataset_name]["versions"][-1]
    return f"gs://gresearch/robotics/{dataset_name}/{version}"
