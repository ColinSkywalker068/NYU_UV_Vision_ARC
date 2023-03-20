import time
import serial

print("We are live \n")

ser = serial.Serial(
        port = "/dev/ttyTHS1",
        baudrate = 115200,
        bytesize = serial.EIGHTBITS,
        parity = serial.PARITY_NONE,
        stopbits = serial.STOPBITS_ONE,
        timeout=1
        )

time.sleep(1)

while True:
    time.sleep(1)
    data = input("Type what needs to be sent: ")
    ser.write(data.encode())
    ser.write("\n".encode())
    
    while ser.inWaiting()==0:
        time.sleep(0.01)
    data = ser.readline()
    print("receiving", data.decode('ascii'))

