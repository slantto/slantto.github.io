#include <Arduino.h>

#include <SoftwareSerial.h>
#include <Wire.h>
#define rxPin 9  // pin 9 connects to smcSerial RX  (not used in this example)
#define txPin 10  // pin 10 connects to smcSerial TX
#define xstepPin 3
#define xdirPin 4
#define ystepPin 5
#define ydirPin 6
#define zstepPin 7
#define zdirPin 8
SoftwareSerial smcSerial = SoftwareSerial(rxPin, txPin);

float currentX=0;
float currentY=0;
float currentZ=0;
float revx;
float revy;
float revz;
float x;
float y;
float z;



// required to allow motors to move
// must be called when controller restarts and after any error
void exitSafeStart()
{
  smcSerial.write(0x83);
}

// speed should be a number from -3200 to 3200
void setMotorSpeed(int speed)
{
  if (speed < 0)
  {
    smcSerial.write(0x86);  // motor reverse command
    speed = -speed;  // make speed positive
  }
  else
  {
    smcSerial.write(0x85);  // motor forward command
  }
  smcSerial.write(speed & 0x1F);
  smcSerial.write(speed >> 5);
}


void setup() {
   // initialize software serial object with baud rate of 19.2 kbps
  smcSerial.begin(19200);

  // the Simple Motor Controller must be running for at least 1 ms
   delay(5);


  smcSerial.write(0xAA);  // send baud-indicator byte

  // Sets the two pins as Outputs
  pinMode(xstepPin,OUTPUT);
  pinMode(xdirPin,OUTPUT);
  pinMode(ystepPin, OUTPUT);
  pinMode(ydirPin,OUTPUT);
  pinMode(zstepPin, OUTPUT);
  pinMode(zdirPin,OUTPUT);
  Serial.begin(9600);

  exitSafeStart();  // clear the safe-start violation and let the motor run

}
void loop() {

  Serial.println("Enter number of mm for x(whole numbers only!)");
  while (Serial.available()==0){ }
   x = Serial.parseFloat();

  Serial.println("Enter number of mm for y(whole numbers only!)");
  while (Serial.available()==0){ }
   y = Serial.parseFloat();

  DrillXYPos(x,y);


    currentX = x;
    currentY = y;
Serial.println("DONE!");

//Z position and drill speed

  Serial.println("Input drill depth in mm(0 means return to top)");
  while (Serial.available()==0){ }
  float z = Serial.parseFloat();

  Serial.println("Input drill speed in from 1 to 3200");
  while (Serial.available()==0){ }
  int s = Serial.parseInt();

  DrillZPos(z, s);

  setMotorSpeed(0);
  currentZ = z;

  Serial.println("DONE!");
  delay(1000); // One second delay


}
