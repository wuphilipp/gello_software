import os
import os.path as osp

DATA_DIR = os.environ["DATA_DIR"]
DATASET_DIR = os.environ["DATASET_DIR"]

BC_DATASET = {
    "place_bag": {
        "train_dataset": osp.join(DATASET_DIR, "place_bag_train"),
        "val_dataset_l1": osp.join(DATASET_DIR, "place_bag_val_l1"),
        "val_dataset_l2": osp.join(DATASET_DIR, "place_bag_val_l2"),
    },
}

ISAACGYM_TASK = {
    "open_drawer": {
        "expert_policy": osp.join(
            DATA_DIR, "yiran_pretrained/open_drawer/model_15400.pt"
        ),
        "task": "OneFrankaCabinetPCPartialCPMap",
        "task_config": "cfg/open_drawer_expert.yaml",
        "algo_config": "cfg/ppo_pc_pure/config.yaml",
        "algo": "ppo_pc_pure",  # Algo used to train the expert policy
        "train_iteration": 10,
        "save_freq": 1,
    },
    "open_door_21": {
        "expert_policy": osp.join(
            DATA_DIR, "yiran_pretrained/open_door/model_16400.pt"
        ),
        "task": "OneFrankaCabinetPCPartialCPMap",
        "task_config": "cfg/open_door_expert.yaml",
        "algo_config": "cfg/ppo_pc_pure/config.yaml",
        "algo": "ppo_pc_pure",
        "train_iteration": 20,
        "save_freq": 5,
    },
}
