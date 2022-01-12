import serial

#!python -m serial.tools.list_ports

def make_cmd(command):
    if command:
        crc = 0
        for b in command:
            crc = crc ^ b
        return command + b'*' + ('%02X' % crc).encode() + b'\r\n'
    return b''

try:
    ahrs = serial.Serial('COM4', 115200, timeout=1)
    if ahrs.is_open:
        try:
            ahrs.reset_input_buffer()
            ahrs.write(make_cmd(b'@mode,AT'))
            ahrs.read_until()
            ahrs.write(make_cmd(b'@asc_out,QUAT'))
            ahrs.read_until()
            ahrs.write(make_cmd(b'@divider,1'))
            ahrs.read_until()
            ahrs.write(make_cmd(b'@mode,AC'))
            ahrs.read_until()
            while True:
                buffer = ahrs.read_until()
                packet = buffer.split(b',')
                if len(packet) == 6:
                    seq = int(packet[1])
                    qx = float(packet[2])
                    qy = float(packet[3])
                    qz = float(packet[4])
                    qw = float(packet[5].split(b'*')[0])
                    print(f'XYZW: {qx:.2f}, {qy:.2f}, {qz:.2f}, {qw:.2f}')
                ahrs.reset_input_buffer()
        except:
            ahrs.close()
except:
    print('Cannot open the port')