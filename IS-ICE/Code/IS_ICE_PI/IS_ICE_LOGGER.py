#!/usr/bin/env python3
"""
THIS IS THE IS-ICE COMMUNICATION AND CONTROL SOFTWARE ON THE RASPBERRY PI 3

SENDS COMMANDS TO THE ARDUINO AND RECEIVES SENSOR DATA FOR DISPLAY AND LOGGING
"""
import serial
#import time
import datetime



def sendToArduino(sendStr):
  ser.write(sendStr)

def recvFromArduino():
    global startMarker, endMarker
    # print("recieving reply")
    ck = ""
    x = "z" # any value that is not an end- or startMarker
    byteCount = -1 # to allow for the fact that the last increment will be one too many

    # wait for the start character
    while  ord(x) != startMarker:
        x = ser.read()

    print("recieving reply")

  # save data until the end marker is found
    while ord(x) != endMarker:
      if ord(x) != startMarker:
        ck = ck + str(x)
        byteCount += 1
      x = ser.read()
      x = x.decode()
    # ck = ck.decode()
    return(ck)

def waitForArduino():

    # wait until the Arduino sends 'Arduino Ready' - allows time for Arduino reset
    # it also ensures that any bytes left over from a previous message are discarded

    global startMarker, endMarker

    msg = ""
    while msg.find("Arduino is ready") == -1:

        #while ser.inWaiting()== 0:
         #   pass

        msg = recvFromArduino()
        print(msg)

    print(msg)


if __name__ == '__main__':
    # NOTE the user must ensure that the serial port and baudrate are correct
    serPort = "/dev/ttyACM0"
    baudRate = 9600
    ser = serial.Serial(serPort, baudRate)
    print("Serial port " + serPort + " opened  Baudrate " + str(baudRate))
    ser.flushInput()

    #Start = <, End = >
    startMarker = 60
    endMarker = 62

    #open log txt files
    DAY2Log = open("DAY2Log.txt", "w")


    #Wait for arduino to run through startup
    waitForArduino()
    hist = "Begin Log " + str(datetime.datetime.now()) + "\n"
    print(hist)
    DAY2Log.write(hist)
    DAY2Log.flush()

    msg = ""
    while msg.find("SHUTDOWN COMPLETE") == -1:
        msg = recvFromArduino()
        hist = str(datetime.datetime.now()) + " " + msg + "\n"
        print(hist)
        DAY2Log.write(hist)
        DAY2Log.flush()
    print(msg)

    ser.close()
    DAY2Log.close()