import socket
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import xgboost as xgb
from scipy.interpolate import make_interp_spline, BSpline

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
    recording_time = 80

    # Initialise prediction variable in which you want to place the predicted output of your classifier
    prediction = np.zeros(recording_time * CHUNK_SIZE)

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

    # rms_of_all_channels = []
    # for channel in tqdm(range(channel_data.shape[0])):
    #     rms_of_a_channel = []
    #     for i in range(len(channel_data[channel])):
    #         window = channel_data[channel][i: i + window_samples].astype(int)
    #         window_average = np.sqrt(abs(np.sum(np.square(window)) / window_samples))
    #         rms_of_a_channel.append(window_average)
    #     rms_of_all_channels.append(rms_of_a_channel)

    rms_of_all_channels = np.array(channel_data)
    test_data = rms_of_all_channels[0:320, :].T
    test_data = xgb.DMatrix(data=test_data, label=None)
    prediction = model_xgb.predict(test_data)
    for i in range(len(prediction)):
        window = prediction[i: i + window_samples].astype(int)
        window_average = np.sqrt(abs(np.sum(np.square(window)) / window_samples))
        prediction[i] = window_average

    # x_y_spline = make_interp_spline(range(len(prediction)), prediction, k=5)
    # x_new = np.linspace(0, len(prediction) - 1, len(prediction))
    # y_new = x_y_spline(x_new)
    # plt.plot(x_new, y_new)
    plt.plot(prediction)
    plt.show()


if __name__ == "__main__":
    model_xgb = xgb.Booster()
    model_xgb.load_model("model.json")
    main(model_xgb)
