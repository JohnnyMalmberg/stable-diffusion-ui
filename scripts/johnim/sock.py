import socket
import os
import time
import os.path
from collections import deque
from queue import Queue

def listener(state, on_command):
    sock_path = "/tmp/sd_command.s"

    if os.path.exists(sock_path):
        os.remove(sock_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    server.bind(sock_path)

    server.settimeout(5)

    while state.running:
        try:
            server.listen(1)
            conn, addr = server.accept()

            data = conn.recv(1024)

            if data:
                command = data.decode('ascii')
                on_command(command)

        except Exception as e:
            if not state.debug_quiet:
                print(f'[Listener] {e}')
            continue
    
    print('[Listener] Done.')

# Negative codes will cause errors!
CODE_EXIT = 69420
CODE_COMMAND = 3
CODE_IMAGE_RESULT = 1

def fake_transmitter(state):
    q = state.transmission_queue
    while state.running:
        try:
            while not q.empty():
                with q.mutex:
                    q.clear()
            time.sleep(3)
        except:
            pass


def establish_transmitter(state):
    q = state.transmission_queue

    sock_path = "/tmp/sd_transmitter.s"

    if os.path.exists(sock_path):
        os.remove(sock_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    server.bind(sock_path)

    server.settimeout(5)

    while state.running:
        try:
            server.listen(1)
            conn, addr = server.accept()

            while state.running: # Exeptions break us free of the loop
                while not q.empty():
                    (code, content) = q.get()
                    conn.sendall(code.to_bytes(4, 'little'))
                    conn.sendall(len(content).to_bytes(4, 'little'))
                    conn.sendall(content)
                time.sleep(1)
            
            conn.sendall(CODE_EXIT.to_bytes(4, 'little'))

            conn.close()
        except Exception as e:
            if not state.debug_quiet:
                print(f'[Transmitter] {e}')
            # "timed out" appears regularly in the console when a client is not connected. this is normal
            # "[Errno 32] Broken pipe" appears when a client disconnects. this is normal
            continue
    
    print('[Transmitter] Done.')

