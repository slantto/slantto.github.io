#!/usr/bin/env python
"""
THIS IS THE IS-ICE COMMUNICATION AND CONTROL SOFTWARE ON THE RASPBERRY PI 3

SENDS COMMANDS TO THE ARDUINO AND RECEIVES SENSOR DATA FOR DISPLAY AND LOGGING
"""

def sendToArduino(sendStr):
  ser.write(sendStr)
def waitforard

def returnToPI
    ardresp = ser.readline
    # print(ardresp)
    return ardresp
def horzmove



#import tkinter as tk
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
    serPort = "/dev/ttyS80"
    baudRate = 9600
    ser = serial.Serial(serPort, baudRate)
    print("Serial port " + serPort + " opened  Baudrate " + str(baudRate))
    PowerLog = open("PowerLog.txt","w")
    WOBLog = open("WOBLog.txt","w")
    COMHist = open("COMHist.txt","w")
   # logger = myThread(1, "logger")
   # inputer = myThread(2, "inputer")

    while 1:
        ardcom = input("Please input command number(h for help):")

        if ardcom == "h":
            print("1 = move x and z, 2 = move y, 3 = drill operation, 4 = meltchamber operaton, 5 =  robot state info, X for exit loop")
        elif ardcom == "secret":
            print("Welcome to the secret menu, please select a command....")
        elif ardcom == "1":
            ser.write(ardcom)
            ardresp = ser.readline
            print(ardresp)
            COMHist.write(ardresp)
            ardcom = input("Move X in mm")
            ser.write(ardcom)
            ardresp = ser.readline
            print(ardresp)
            COMHist.write(ardresp)
            ardcom = input("Move Z in mm")
            ser.write(ardcom)
            msg = ""
            while msg.find("done") == -1:
                msg = returnToPI()
                COMHist.write(msg)
                print(msg)
        elif ardcom == "2":
            ser.write(ardcom)
            ardresp = ser.readline
            print(ardresp)
            COMHist.write(ardresp)
            ardcom = input("Move Y in mm")
            ser.write(ardcom)
            ardresp = ser.readline
            print(ardresp)
            COMHist.write(ardresp)
            ardcom = input("Drill speed 0 to 255")
            ser.write(ardcom)
            msg = ""
            while msg.find("done") == -1:
                msg = returnToPI()
                COMHist.write(msg)
                print(msg)
        elif ardcom == "3":
            ser.write(ardcom)
            ardresp = ser.readline
            print(ardresp)
            COMHist.write(ardresp)
            ardcom = input("Move X in mm")
            ser.write(ardcom)
            ardresp = ser.readline
            print(ardresp)
            COMHist.write(ardresp)
            ardcom = input("Move Z in mm")
            ser.write(ardcom)
            ardresp = ser.readline
            print(ardresp)
            ardcom = input("Melt time in minutes(recommend no more than 5 minutes)")
            ser.write(ardcom)
            msg = ""
            while msg.find("done") == -1:
                msg = returnToPI()
                COMHist.write(msg)
                print(msg)
        elif ardcom == "4":
            ser.write(ardcom)
            ardresp = ser.readline
            print(ardresp)
            COMHist.write(ardresp)
        elif ardcom == "5":
            ser.write(ardcom)
            msg = ""
            while msg.find("done") == -1:
                msg = returnToPI()
                COMHist.write(msg)
                print(msg)
        elif ardcom == "X":
            print("Exiting")
            break
        else:
            print("BAD COMMAND, TRY AGAIN")






