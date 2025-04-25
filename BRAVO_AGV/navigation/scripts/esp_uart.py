import serial
from time import sleep

ser = serial.Serial("/dev/ttyAMA0", 9600)
print("Configuration ready")
print(ser.name)
value1 = 10.51
value = str(value1)
newline = "\n"
i = 0
while True:
	#received_date = ser.read()
	#sleep(0.03)
	#data_left = ser.inWaiting()
	#received_data += ser.read(data_left)
	#print(received_data)
	#ser.write(b'hello F \n')
	ser.write(b'1n')
	ser.write((value.encode()+newline.encode()))
	sleep(0.03)
	i+=1
	if (i == 10):
		break
print("closing")
ser.close()

