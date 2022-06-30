import math
import socket
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import xgboost as xgb
from scipy.signal import lfilter
from scipy.signal import savgol_filter

# Header for receiving activation response from the EMG device
EMG_HEADER = 8

# Server address properties
IP = "localhost"
PORT = 31000

"""Streaming properties:
        FREAD: int
            The amount of chunks send per second.
        SAMPLING_FREQUENCY: int
            Sampling frequency of the EMG device.
        CHUNK_SIZE: int
            Sample size of the received EMG chunk
        BUFFER_SIZE: int
            The amount of Bytes you want to receive in each frame.    
"""
FREAD = 8
SAMPLING_FREQUENCY = 2048
CHUNK_SIZE = int(SAMPLING_FREQUENCY / FREAD)
BUFFER_SIZE = 408 * CHUNK_SIZE * 2

# Define the task you want to record: e.g., Test, Fist, etc.
TASK = ""


def main(model_xgb):
    # Open EMG socket and connect it to the EMG device
    emg_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    emg_socket.connect((IP, PORT))

    # Send message to start streaming
    emg_socket.send("startTX".encode("utf-8"))

    # Print response of the EMG device
    print(f"Connection answer by EMG: {emg_socket.recv(EMG_HEADER).decode('utf-8')}")

    # TODO: Define the recording time
    recording_time = 60

    # TODO: Write a loop where you receive the chunks from the EMG simulator. Slice the data in the correct shape
    #  of 320 channels (electrode grids number 0, 2, 3, 4, 5). Use the sliced chunk as input to your model and let it
    #  classify the input to the correct task. Show the output of the prediction as a feedback to e.g., a patient. You
    #  can keep it simple (plot, etc.)!
    #  After the recording plot the prediction signal and save it in the results/ folder.

    signal = np.frombuffer(emg_socket.recv(BUFFER_SIZE), dtype=np.int16).reshape((408, -1), order="F")
    for _ in tqdm(range(1, recording_time * FREAD)):
        message = np.frombuffer(emg_socket.recv(BUFFER_SIZE), dtype=np.int16).reshape((408, -1), order="F")
        signal = np.append(signal, message, 1)

    window_size = 0.25
    window_samples = round(window_size * SAMPLING_FREQUENCY)  # Convert window size from time into sample number

    first_batch = signal[0:64, :]  # Cover first 64 EMG channels
    second_batch = signal[128:384, :]  # Cover last 256 EMG channels
    channel_data = np.concatenate((first_batch, second_batch), axis=0)

    rms_of_all_channels = np.array(channel_data)
    test_data = rms_of_all_channels[0:320, :].T
    test_data = xgb.DMatrix(data=test_data, label=None)
    prediction = model_xgb.predict(test_data)
    for i in range(len(prediction)):
        window = prediction[i: i + window_samples].astype(int)
        window_average = np.sqrt(abs(np.sum(np.square(window)) / window_samples))
        prediction[i] = window_average

    time = np.arange(0, len(prediction) / SAMPLING_FREQUENCY, 1 / SAMPLING_FREQUENCY)
    yy = savgol_filter(prediction, 101, 2)
    plt.plot(time, yy)
    plt.xlabel("Time [s]")
    plt.ylabel("Electrode")
    plt.title("Prediction as a feedback from the patient")
    plt.show()


if __name__ == "__main__":
    model_xgb = xgb.Booster()
    model_xgb.load_model("model_4.json")
    main(model_xgb)
