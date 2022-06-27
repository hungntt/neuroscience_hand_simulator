import socket

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
TASK = "Test"


def main(plot=False):
    # Open EMG socket and connect it to the EMG device
    print(TASK)
    recording_time = 60 if TASK == "Test" else 10

    emg_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    emg_socket.connect((IP, PORT))

    # Send message to start streaming
    emg_socket.send("startTX".encode("utf-8"))

    # Print response of the EMG device
    print(f"Connection answer by EMG: {emg_socket.recv(EMG_HEADER).decode('utf-8')}")

    # TODO: Write a loop where you receive the chunks from the EMG simulator. Calculate the mean of the moving average
    #  RMS with a time window of 0.125 s in real-time and place the calculated chunk in the rms variable. After that
    #  update your plot.
    #  Use "message = np.frombuffer(emg_socket.recv(BUFFER_SIZE), dtype=np.int16).reshape((408, -1), order="F")" to
    #  receive the chunk. Save the recorded data and the labels in the case of the Test dataset
    #  in the train_data/ or test_data/ folder after the recording stopped. Save the calculated RMS signals into the
    #  results/ folder.

    signal = np.frombuffer(emg_socket.recv(BUFFER_SIZE), dtype=np.int16).reshape((408, -1), order="F")

    for _ in tqdm(range(1, recording_time * FREAD)):
        message = np.frombuffer(emg_socket.recv(BUFFER_SIZE), dtype=np.int16).reshape((408, -1), order="F")
        signal = np.append(signal, message, 1)

    # Define window size
    window_size = 0.125
    window_samples = round(window_size * SAMPLING_FREQUENCY)  # Convert window size from time into sample number

    first_batch = signal[0:64, :]  # Cover first 64 EMG channels
    second_batch = signal[128:384, :]  # Cover last 256 EMG channels
    channel_data = np.concatenate((first_batch, second_batch), axis=0)  # Concatenate the two batches

    if plot:
        rms_of_all_channels = np.array([])
        print('Computing the RMS')
        for i in tqdm(range(len(channel_data[0]))):
            window = channel_data.T[i: i + window_samples].T  # Select limited number of samples for all channels
            window = window.astype(int)  # Convert values to integer before applying Square-Root to avoid errors
            window_average = np.sqrt(abs(np.sum(np.square(window), axis=1) / window_samples))  # RMS formula
            if plot:
                window_average = np.mean(window_average)  # Compute average of RMS values over all channels
            rms_of_all_channels = np.append(rms_of_all_channels, window_average)
    else:
        rms_of_all_channels = []
        for channel in tqdm(range(channel_data.shape[0])):
            rms_of_a_channel = []
            for i in range(len(channel_data[channel])):
                window = channel_data[channel][i: i + window_samples].astype(int)
                window_average = np.sqrt(abs(np.sum(np.square(window)) / window_samples))
                rms_of_a_channel.append(window_average)
            rms_of_all_channels.append(rms_of_a_channel)

    if TASK == "Test":
        print("Adding the label")
        label_testset = np.expand_dims(signal[400, :], axis=0).squeeze()  # Cover the label of the testset
        rms_of_all_channels.append(label_testset)  # Concatenate label with the channels

    rms_of_all_channels = np.array(rms_of_all_channels)

    print("Saving the dataset")
    np.save(f"./test_data/{TASK}.npy", rms_of_all_channels) if TASK == "Test" \
        else np.save(f"./train_data/{TASK}.npy", rms_of_all_channels)

    if plot:
        plt.figure(1)
        plt.clf()
        plt.plot(channel_data[0, :].T)  # choose here how many channels to plot
        # plt.plot(rms)
        plt.plot(np.load(f"./train_data/{TASK}.npy"))
        plt.xlabel('Samples')
        plt.ylabel('EMG')
        plt.title('Average RMS')
        plt.grid(True)

        # Plot data in realtime
        plt.figure(2)
        i = 0
        time = np.arange(0, len(rms) / SAMPLING_FREQUENCY, 1 / SAMPLING_FREQUENCY)
        while i < len(rms):
            plt.cla()
            plt.plot(time[i:i + 1000], rms[i:i + 1000])
            plt.xlabel('Samples')
            plt.ylabel('EMG')
            plt.title('Realtime plotting RMS')
            plt.grid(True)
            plt.pause(1 / SAMPLING_FREQUENCY)
            i += 20

        plt.show()


if __name__ == '__main__':
    main()
