import socket
import select
import time
import argparse

from jsonschema._validators import required

from emg_datasets import TrainData, TestData


# Class for EMG simulator. DON'T CHANGE ANYTHING HERE
class EMGSimulator:
    """Class simulates an EMG device.

    Attributes
    ----------
    tcp_ip : string
    tcp_port: int
    server_socket: socket.socket object
    sockets_list: list
    read_buffer: int
    """

    def __init__(self, tcp_ip="localhost", tcp_port=31000, fread=8, emg_data=None):
        # Socket settings
        self.tcp_ip = tcp_ip
        self.tcp_port = tcp_port
        self.server_socket = None
        self.sockets_list = []
        self.read_buffer = 10

        # Streaming characteristics
        self.fread = fread
        self.sampling_frequency = 2048
        self.frame_len = self.sampling_frequency / self.fread
        self.frame_time = 1 / self.fread
        self.frame_counter = 0

        self.start_command = "startTX"
        self.stop_command = "stopTX"
        self.feedback_msg = "OTBiolab".encode("utf-8")
        self.client_connected = False

        # data buffer
        self.emg_data = emg_data

    def open_connection(self):
        """Opens the server connection.

        The server socket gets opened as a TCP/IP socket and bound to the given IP and Port. The server is also
        set to listen to incoming data and blocks the code until data is arrived. Append server socket to
        sockets list.
        """
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.tcp_ip, self.tcp_port))
        self.server_socket.listen()
        self.server_socket.setblocking(True)

        self.sockets_list.append(self.server_socket)

    def close_connection(self):
        """Close server socket connection."""
        self.server_socket.close()

    def start_stream(self):
        """Start the stream of incoming and outgoing data."""
        self.frame_counter = 0
        last_frame = time.time()

        while True:
            read_sockets, write_sockets, exception_sockets = select.select(
                    self.sockets_list, self.sockets_list[1:], self.sockets_list
            )
            for notified_socket in read_sockets:
                if notified_socket == self.server_socket:
                    try:
                        client_socket, client_address = self.server_socket.accept()
                        message = client_socket.recv(self.read_buffer)

                        if not len(message):
                            continue

                        self.sockets_list.append(client_socket)
                        print(
                                f"Accepted new connection from {client_address[0]}: {client_address[1]}!"
                        )
                        client_socket.send(self.feedback_msg)

                    except ConnectionAbortedError:
                        continue

                    except ConnectionResetError:
                        continue

                else:
                    try:
                        message = notified_socket.recv(self.read_buffer)
                        if not len(message):
                            self.sockets_list.remove(notified_socket)
                            continue
                        if message.decode("utf-8") == self.stop_command:
                            if notified_socket in self.sockets_list:
                                self.sockets_list.remove(notified_socket)
                            continue

                    except ConnectionAbortedError:
                        if notified_socket in self.sockets_list:
                            self.sockets_list.remove(notified_socket)
                        continue

                    except ConnectionResetError:
                        if notified_socket in self.sockets_list:
                            self.sockets_list.remove(notified_socket)
                        continue

            while time.time() - last_frame < self.frame_time:
                continue

            if self.frame_counter * self.frame_len == self.emg_data.shape[1]:
                self.frame_counter = 0
            index1 = int(self.frame_counter * self.frame_len)
            index2 = int(self.frame_counter * self.frame_len + self.frame_len)
            data_chunk = self.emg_data[:, index1:index2]

            data_to_send = data_chunk.tobytes(order="F")
            for notified_socket in write_sockets:
                try:
                    notified_socket.send(data_to_send)
                    print(data_chunk.shape)
                    print(f"Time since last frame: {1 / (time.time() - last_frame)}")
                    last_frame = time.time()
                    self.frame_counter += 1

                except ConnectionAbortedError:
                    if notified_socket in self.sockets_list:
                        self.sockets_list.remove(notified_socket)
                    continue

                except ConnectionResetError:
                    if notified_socket in self.sockets_list:
                        self.sockets_list.remove(notified_socket)
                    continue


def main(args):
    dataset = args.data
    print(f"Starting EMG simulator on {dataset}")
    if dataset == "Test":
        emg_dataset = TestData().test_emg["Test"]
    elif dataset == "Fist":
        emg_dataset = TrainData().fist_emg["Fist"]
    elif dataset == "Thumb":
        emg_dataset = TrainData().thumb_emg["Thumb"]
    elif dataset == "Index":
        emg_dataset = TrainData().index_emg["Index"]
    elif dataset == "Middle":
        emg_dataset = TrainData().middle_emg["Middle"]
    elif dataset == "Ring":
        emg_dataset = TrainData().ring_emg["Ring"]
    elif dataset == "Pinky":
        emg_dataset = TrainData().pinky_emg["Pinky"]

    # DON'T CHANGE ANYTHING HERE
    emg_socket = EMGSimulator(
            tcp_ip="localhost", tcp_port=31000, fread=8, emg_data=emg_dataset
    )
    emg_socket.open_connection()
    print("Server open for connections!")
    emg_socket.start_stream()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Data to be used for the simulation', default="Fist")
    args = parser.parse_args()
    main(args)
