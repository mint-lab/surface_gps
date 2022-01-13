import numpy as np
import serial, threading, time
from scipy.spatial.transform import Rotation
import pyvista as pv

rad2deg = lambda rad: rad * 180 / np.pi
deg2rad = lambda deg: deg * np.pi / 180

class myAHRSPlus:
    def __init__(self, verbose=True):
        self.__dev = serial.Serial()
        self.seq = 0
        self.xyzw = [0., 0., 0., 1.]
        self.__verbose = verbose
        self.__lock = threading.Lock()
        self.__thread = threading.Thread(target=self.recv_data)
        self.__thread_exit = False

    def __del__(self):
        self.close()

    def open(self, port, baudrate=115200, timeout=1):
        if not self.__dev.is_open:
            self.__dev.port = port
            self.__dev.baudrate = baudrate
            self.__dev.timeout = timeout
            self.__dev.open()
            if self.__dev.is_open:
                self.__dev.reset_input_buffer()
                self.send_cmd(b'@mode,AT')
                self.send_cmd(b'@asc_out,QUAT')
                self.send_cmd(b'@divider,1')
                self.send_cmd(b'@mode,AC')
                self.__thread.start()
                return True
            else:
                if self.__verbose:
                    print('[myAHRSPlus] Cannot open the port, {port}')
        else:
            if self.__verbose:
                print('[myAHRSPlus] The port, {port}, is already open.')
        return False

    def is_open(self):
        return self.__dev.is_open

    def close(self, timeout=10):
        self.__thread_exit = True
        for trial in range(1, 100):
            if not self.__thread.is_alive():
                break
            time.sleep(timeout / 100.)
        self.__dev.close()

    def get_xyzw(self):
        with self.__lock:
            return list(self.xyzw) # Return the copy

    def send_cmd(self, command):
        if command:
            crc = 0
            for b in command:
                crc = crc ^ b
            packet = command + b'*' + ('%02X' % crc).encode() + b'\r\n'
            self.__dev.write(packet)
            self.__dev.read_until()
        return False

    def recv_data(self):
        self.__thread_exit = False
        if self.__verbose:
            print("[myAHRSPlus] Start 'recv_data' thread.")
        while not self.__thread_exit:
            packet = self.__dev.read_until()
            token = packet.split(b',')
            if len(token) == 6:
                with self.__lock:
                    self.seq = int(token[1])
                    self.xyzw = [float(token[2]), float(token[3]), float(token[4]), float(token[5].split(b'*')[0])]
            else:
                if self.__verbose:
                    print(f'[myAHRSPlus] Receive an wrong packet, {str(packet)}')
            self.__dev.reset_input_buffer()
        print("[myAHRSPlus] Terminate 'recv_data' thread.")



if __name__ == '__main__':
    ahrs = myAHRSPlus()
    if ahrs.open('COM4'):
        plotter = pv.Plotter()
        plotter.add_axes_at_origin('r', 'g', 'b')
        ahrs_mesh = pv.Cube(x_length=0.8, y_length=0.4, z_length=0.004)
        ahrs_actor = plotter.add_mesh(ahrs_mesh, color='r')
        plotter.set_viewup([0, 0, 1])
        plotter.show(title='SeoulTech AHRS Visualization', interactive_update=True)

        try:
            while True:
                q_xyzw = ahrs.get_xyzw()
                y, x, z = Rotation.from_quat(q_xyzw).as_euler('yxz', degrees=True)
                ahrs_actor.SetOrientation(x, y, z)
                plotter.update()
        except KeyboardInterrupt:
            pass

        plotter.close()
        ahrs.close()