
import nanpy import (ArduinoApi, SerialManager)
import time

DrillSpeedPin = 65
DrillDirPin = 59

XSTEPPIN  = 54
XDIRPIN = 55
XENPIN = 38
XMINPIN = 3
XMAXPIN = 2

YSTEPPIN = 60
YDIRPIN = 61
YENPIN = 56
YMINPIN = 14
YMAXPIN = 15

ZSTEPPIN = 46
ZDIRPIN = 48
ZENPIN = 56
ZMINPIN = 18
ZMAXPIN = 19

SDPOWER = -1
SDSS = 53
LEDPIN = 13

FANPIN = 9
CSENSEPIN = 69

PSONPIN = 12
KILLPIN = -1
HEATER0PIN = 10
HEATER1PIN = 8
TEMP0PIN = 67
TEMP1PIN = 68

DOUT


try:
    connection = SerialManager()
    a = ArduinoApi(connection = connection)
except:
    print("connection failed")

