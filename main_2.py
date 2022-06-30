import argparse
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


def main(args):
    # Open EMG socket and connect it to the EMG device
    TASK = args.data
    rms = args.rms
    plot = args.plot
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

    if TASK == "Test":
        print("Adding the label")
        label_testset = signal[400, :],  # Cover the label of the testset
        rms_of_all_channels = np.concatenate((channel_data, label_testset), axis=0)
    else:
        rms_of_all_channels = channel_data

    if rms:
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

    # Save the RMS signals into the results/ folder
    print('Saving the dataset')
    np.save(f'./{"test" if TASK == "Test" else "train"}_data/{TASK}.npy', rms_of_all_channels) if plot else \
        np.save(f'./original_{"test" if TASK == "Test" else "train"}_data/{TASK}.npy', rms_of_all_channels)

    if plot:
        time = np.arange(0, len(rms_of_all_channels) / SAMPLING_FREQUENCY, 1 / SAMPLING_FREQUENCY)
        plt.figure(1)
        plt.clf()
        plt.plot(time, channel_data[0, :].T)  # choose here how many channels to plot
        # plt.plot(rms)
        plt.plot(np.load(f'./{"test" if TASK == "Test" else "train"}_data/{TASK}.npy'))
        plt.xlabel('Times [s]')
        plt.ylabel('EMG')
        plt.title(f'Average RMS on ')
        plt.grid(True)

        # Plot data in realtime
        plt.figure(2)
        i = 0
        while i < len(rms_of_all_channels):
            plt.cla()
            plt.plot(time[i:i + 1000], rms_of_all_channels[i:i + 1000])
            plt.xlabel('Times [s]')
            plt.ylabel('EMG')
            plt.title(f'Real-time plotting {TASK} RMS')
            plt.grid(True)
            plt.pause(1 / SAMPLING_FREQUENCY)
            i += 20

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Data to be used for the simulation', default="Fist")
    parser.add_argument('--plot', action='store_true', help='Plot the data in realtime')
    parser.add_argument('--rms', action='store_true', help='Calculate the RMS of the data')
    args = parser.parse_args()
    main(args=args)
