import serial
import time
import threading

#class myThread(threading.Thread):
 #   def __init__(self, threadID, name):
 #   threading.Thread.__init__(self)
#    self.threadID = threadID
#    self.name = name


if __name__ == '__main__':
    # NOTE the user must ensure that the serial port and baudrate are correct
    serPort = "/dev/ttyACM0"
    baudRate = 9600
    ser = serial.Serial(serPort, baudRate)
    print("Serial port " + serPort + " opened  Baudrate " + str(baudRate))
    ser.flushInput()

    PowerLog = open("PowerLog.txt", "w")
    WOBLog = open("WOBLog.txt", "w")
    COMHist = open("COMHist.txt", "w")
   # logger = myThread(1, "logger")
   # inputer = myThread(2, "inputer")
    while 1:
        if (ser.inWaiting()>0):
            data = ser.readline()
            print(data)


