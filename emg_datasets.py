import glob
import os
import numpy as np
from scipy.io import loadmat


# IMPORTANT: DON'T CHANGE ANYTHING HERE


def create_simulation_data(data, chs=(0, 2, 3, 4, 5)):
    sliced_datas = {}

    for i in range(len(chs)):
        sliced_datas[chs[i]] = data[i * 64: i * 64 + 64]

    empty_channel = np.zeros((64, data.shape[1]))
    last_channels = np.zeros((24, data.shape[1]))
    whole_sliced_data = []

    for i in range(6):
        if i in chs:
            whole_sliced_data.append(sliced_datas[i])
        else:
            whole_sliced_data.append(empty_channel)

    whole_sliced_data.append(last_channels)
    data_full_channels = np.concatenate(whole_sliced_data)
    mod = data_full_channels.shape[1] % 2048
    if mod == 0:
        mod = 2048
    missing_data = 2048 - mod
    emg_simulated = np.concatenate(
            (data_full_channels, np.zeros((408, missing_data))), axis=-1
    )

    return np.array(emg_simulated, dtype=np.int16)


def get_files_mat(basic_path, keyword):
    file_paths = []
    for path in glob.glob(os.path.join(basic_path, f"{keyword}.mat")):
        file_paths.append(path)

    files = {}
    for path in file_paths:
        mat = loadmat(path)
        files[keyword[:-1]] = create_simulation_data(mat["EMG_SYNC"])

    return files


def get_files_npy(basic_path, keyword):
    file_paths = []
    for path in glob.glob(os.path.join(basic_path, f"{keyword}.npy")):
        file_paths.append(path)

    files = {}
    for path in file_paths:
        emg = np.load(path)
        files[keyword] = create_simulation_data(emg)

    return files


class TrainData:
    def __init__(self):
        self.basic_path = "./data/"
        self.fist_emg = get_files_npy(self.basic_path, "Fist")
        self.thumb_emg = get_files_npy(self.basic_path, "Thumb")
        self.index_emg = get_files_npy(self.basic_path, "Index")
        self.middle_emg = get_files_npy(self.basic_path, "Middle")
        self.ring_emg = get_files_npy(self.basic_path, "Ring")
        self.pinky_emg = get_files_npy(self.basic_path, "Pinky")


class TestData:
    def __init__(self):
        self.basic_path = "./data/"
        self.test_emg = get_files_npy(self.basic_path, "Test")
        label = np.concatenate([np.zeros(10 * 2048) + i for i in range(6)])

        self.test_emg["Test"][400] = label
