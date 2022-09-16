import socket
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
RECORDING_TIME = 20


def main():
    # Open EMG socket and connect it to the EMG device
    emg_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    emg_socket.connect((IP, PORT))

    # Send message to start streaming
    emg_socket.send("startTX".encode("utf-8"))

    # Print response of the EMG device
    print(f"Connection answer by EMG: {emg_socket.recv(EMG_HEADER).decode('utf-8')}")

    # TODO: Write a loop where you receive the chunks from the EMG simulator and plot the first channel after the
    #  recording ends. Use "message = np.frombuffer(emg_socket.recv(BUFFER_SIZE), dtype=np.int16).reshape((408, -1),
    #  order="F")" to receive the chunk. Save the recorded data in the result/ folder. In this task, you need to
    #  save  the streamed data in a list while recording. After the streaming has finished plot the first channel and
    #  save the recorded data in the result/ folder. The recording time is 20 seconds. The EMG streams the data in
    #  chunks whose size depends on the sampling frequency and the streaming  refresh  rate  of  the  device. For
    #  your exercise, the sampling frequency is 2048 Hz, and the refresh rate is 8 Hz. From the sampling
    #  frequency and refresh rate you can then calculate the size of one chunk.

    recording_time = 20
    loop_size = recording_time * FREAD  # since a new chunk arrives once every 8th of a second and the recording
    # lasts 20s, then the loop should run 8 × 20 times

    for _ in tqdm(range(loop_size)):
        message = np.frombuffer(emg_socket.recv(BUFFER_SIZE), dtype=np.int16).reshape((408, -1), order="F")
        try:
            signal = np.append(signal, message, axis=1)
        except NameError:
            signal = message

    time = np.arange(0, len(signal[0]) / SAMPLING_FREQUENCY, 1 / SAMPLING_FREQUENCY)
    plt.figure(1)
    plt.clf()
    plt.plot(time, signal[0, :].T)  # choose here how many channels to plot
    plt.xlabel('Time[s]')
    plt.ylabel('EMG')
    plt.title('EMG fist, channel 1')
    plt.grid(True)
    plt.show()
    np.save("./results/chunk_list.npy", signal)


if __name__ == "__main__":
    main()
