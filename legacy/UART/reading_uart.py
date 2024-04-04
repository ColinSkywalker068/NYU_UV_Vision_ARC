import time
import serial


ser = serial.Serial(
    port="/dev/ttyTHS1",
    baudrate=115200,
    parity=serial.PARITY_NONE,
    bytesize=serial.EIGHTBITS,
    timeout=1,
)

ser.write(b"The Code is running\n")
print("We are live \n")
while True:
    # ser.write(str.encode("Hello jetson\n"))
    if ser.inWaiting() == 0:
        pass
    else:
        data = ser.readline()
        print(data)
#        ser.write(data)
# if data =="\r".encode():
#  ser.write("\n".encode())
