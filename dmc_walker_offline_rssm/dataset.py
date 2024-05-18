import os
import numpy as np
from tqdm import tqdm


def load_data(file_path, key):
    with np.load(file_path) as data_file:
        data = data_file[key]
    return data


def get_dataset(path, batch_size, traj_length, precision):
    ds = {}
    data_img_ls = [load_data(path + elem, "image") for elem in tqdm(os.listdir(path))]
    data_action_ls = [
        load_data(path + elem, "action") for elem in tqdm(os.listdir(path))
    ]
    data_reward_ls = [
        load_data(path + elem, "reward") for elem in tqdm(os.listdir(path))
    ]
    data_done_ls = [
        load_data(path + elem, "is_first") for elem in tqdm(os.listdir(path))
    ]
    data_cont_ls = [
        load_data(path + elem, "is_terminal") for elem in tqdm(os.listdir(path))
    ]

    ds["image"] = np.concatenate(data_img_ls).astype(precision)
    ds["action"] = np.concatenate(data_action_ls).astype(precision)
    ds["reward"] = np.concatenate(data_reward_ls).astype(precision)
    ds["is_first"] = np.concatenate(data_done_ls).astype(precision)
    ds["cont"] = 1 - np.concatenate(data_cont_ls).astype(precision)
    num_blocks = len(ds["image"]) // (batch_size * traj_length)
    for k, v in ds.items():
        ds[k] = v[: batch_size * traj_length * num_blocks].reshape(
            -1, batch_size, traj_length, *v.shape[1:]
        )

    return ds


if __name__ == "__main__":
    ds = get_dataset("buffer/", 16, 64, "float16")
    print(ds["image"].shape)
    print(ds["is_first"].shape)