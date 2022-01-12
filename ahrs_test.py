import serial, threading, time

class myAHRSPlus:
    def __init__(self):
        self.__dev = serial.Serial()
        self.seq = 0
        self.xyzw = [0., 0., 0., 1.]
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
                self.__thread_exit = False
                self.__thread.start()
                return True
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
        while not self.__thread_exit:
            packet = self.__dev.read_until()
            token = packet.split(b',')
            if len(token) == 6:
                with self.__lock:
                    self.seq = int(token[1])
                    self.xyzw = [float(token[2]), float(token[3]), float(token[4]), float(token[5].split(b'*')[0])]
            self.__dev.reset_input_buffer()



if __name__ == '__main__':
    ahrs = myAHRSPlus()
    if ahrs.open('COM4'):
        try:
            while True:
                print(f'XYZW: {ahrs.get_xyzw()}')
                time.sleep(0.1)
        except KeyboardInterrupt:
            ahrs.close()