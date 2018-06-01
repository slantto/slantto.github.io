#!/usr/bin/env python3
"""
THIS IS THE IS-ICE COMMUNICATION AND CONTROL SOFTWARE ON THE RASPBERRY PI 3

SENDS COMMANDS TO THE ARDUINO AND RECEIVES SENSOR DATA FOR DISPLAY AND LOGGING
"""

def sendToArduino(sendStr):
  ser.write(sendStr)

def recvFromArduino():
    global startMarker, endMarker

    ck = ""
    x = "z" # any value that is not an end- or startMarker
    byteCount = -1 # to allow for the fact that the last increment will be one too many

    # wait for the start character
    while  ord(x) != startMarker:
        x = ser.read()

  # save data until the end marker is found
    while ord(x) != endMarker:
      if ord(x) != startMarker:
        ck = ck + x
        byteCount += 1
      x = ser.read()

    return(ck)

def waitForArduino():

    # wait until the Arduino sends 'Arduino Ready' - allows time for Arduino reset
    # it also ensures that any bytes left over from a previous message are discarded

    global startMarker, endMarker

    msg = ""
    while msg.find("Arduino is ready") == -1:

        while ser.inWaiting() == 0:
            pass

        msg = recvFromArduino()

        print(msg)

def logFromAduino():
    # Logs and displays sensor data from arduino during drill operations
    global startMarker, endMarker
    msg = ""
    while msg.find("Drill Operation Complete") == -1:
        msg = recvFromArduino()
        print(msg)

#import tkinter as tk
import serial
#import time
import datetime

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
    WPLog = open("WPLog.txt", "w")
    # WOBLog = open("WOBLog.txt", "w")
    COMHist = open("COMHist.txt", "w")

    #Wait for arduino to run through startup
    waitForArduino()

    while 1:
        # Check if any alerts
        if ser.inWaiting() > 0:
            dataRecvd = recvFromArduino()
            print(("Alert Received  " + dataRecvd))
        # Reset Comparam because python is a bitch
        CP1 = 0
        CP2 = CP1
        CP3 = CP2
        # Get Command from user
        command = input("Please input command number(h for help):")
        print(command)

        if command == "h":  # Help
            print("1 = move x and z, 2 = move y, 3 = drill operation, 4 = meltchamber operaton, 5 =  robot state info, X for exit loop")

        elif command == "secret":  # Secret Menu to be completed later
            print("Welcome to the secret menu, please select a command...")

        elif command == "1":   # Move in Horizontal Axis(X and Z)
            Xmm = input("Insert number of mm to move in X")
            Zmm = input("Insert Number of mm to move in Z")
            ardcom = "<" + command + "," + Xmm + "," + Zmm + "," + CP3 + ">"
            print(ardcom)
            hist = str(datetime.datetime.now()) + " " + ardcom
            COMHist.write(hist)
            ser.write(ardcom)

            waitingForReply = True
            while ser.inWaiting() == 0:
                pass
            dataRecvd = recvFromArduino()
            hist = str(datetime.datetime.now()) + " " + dataRecvd
            COMHist.write(hist)
            print(("Reply Received  " + dataRecvd))
            waitingForReply = False

        elif command == "2":  # Move in Vertical Axis (Y)
            Ymm = input("Insert number of mm to move in Y")
            S = input("Drill speed between 0 and 255(max)")
            ardcom = "<" + command + "," + Ymm + "," + S + "," + CP3 + ">"
            print(ardcom)
            hist = str(datetime.datetime.now()) + " " + ardcom
            COMHist.write(hist)
            ser.write(ardcom)

            waitingForReply = True
            while ser.inWaiting() == 0:
                pass
            dataRecvd = recvFromArduino()
            hist = str(datetime.datetime.now()) + " " + dataRecvd
            COMHist.write(hist)
            print(("Reply Received  " + dataRecvd))
            waitingForReply = False

        elif command == "3":  # Drill operation
            Xmm = input("Insert number of mm to move in X")
            Zmm = input("Insert Number of mm to move in Z")
            Mt = input("Insert Number of minutes to leave heater on(recommend no more than 5)")
            ardcom = "<" + command + "," + Xmm + "," + Zmm + "," + Mt + ">"
            print(ardcom)
            hist = str(datetime.datetime.now()) + " " + ardcom
            COMHist.write(hist)
            ser.write(ardcom)

            msg = ""
            while msg.find("Drill Operation Complete") == -1:
                msg = recvFromArduino()
                print(msg)
                wpl = str(datetime.datetime.now()) + " " + msg
                WPLog.write(wpl)



        elif command == "4":  # Heater Manual Switch
            ardcom = "<" + command + "," + CP1 + "," + CP2 + "," + CP3 + ">"
            print(ardcom)
            hist = str(datetime.datetime.now()) + " " + ardcom
            COMHist.write(hist)
            ser.write(ardcom)

            waitingForReply = True
            while ser.inWaiting() == 0:
                pass
            dataRecvd = recvFromArduino()
            hist = str(datetime.datetime.now()) + " " + dataRecvd
            COMHist.write(hist)
            print(("Reply Received  " + dataRecvd))
            waitingForReply = False

        elif command == "5":  # Check RobotState(Kinda useless)
            ardcom = "<" + command + "," + CP1 + "," + CP2 + "," + CP3 + ">"
            print(ardcom)
            hist = str(datetime.datetime.now()) + " " + ardcom
            COMHist.write(hist)
            ser.write(ardcom)

            waitingForReply = True
            while ser.inWaiting() == 0:
                pass
            dataRecvd = recvFromArduino()
            hist = str(datetime.datetime.now()) + " " + dataRecvd
            COMHist.write(hist)
            print(("Reply Received  " + dataRecvd))
            waitingForReply = False

        elif command == "X":
            ser.close
            WPLog.close
            COMHist.close
            print("Exiting")
            break

        elif command == "6":
            print("Have you ever heard the tragedy of Darth Plagueis The Wise...")
            ardcom = "<" + command + "," + CP1 + "," + CP2 + "," + CP3 + ">"
            hist = str(datetime.datetime.now()) + " " + ardcom
            COMHist.write(hist)
            ser.write(ardcom)

            waitingForReply = True
            while ser.inWaiting() == 0:
                pass
            dataRecvd = recvFromArduino()
            hist = str(datetime.datetime.now()) + " " + dataRecvd
            COMHist.write(hist)
            print(("Reply Received  " + dataRecvd))
            waitingForReply = False
        else:
            print("BAD COMMAND, TRY AGAIN")






