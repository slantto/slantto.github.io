
//This is the Motor Control Arduino Code it takes input from the raspberry pi and performs the selected command.

//include libraries
#include <HX711.h>
#include <SoftwareSerial.h>
//#include <Wire.h>
//#include <WireData.h>

//setup I2C
//#define slaveAddress 0x08

//Drill motor pin
#define Drill_Speed_Pin A11 // pin A11(D65)

//Stepper pins
#define X_STEP_PIN         54
#define X_DIR_PIN          55
#define X_ENABLE_PIN       38
#define X_MIN_PIN           3
#define X_MAX_PIN           2

#define Y_STEP_PIN         60
#define Y_DIR_PIN          61
#define Y_ENABLE_PIN       56
#define Y_MIN_PIN          14
#define Y_MAX_PIN          15

#define Z_STEP_PIN         46
#define Z_DIR_PIN          48
#define Z_ENABLE_PIN       62
#define Z_MIN_PIN          18
#define Z_MAX_PIN          19

#define E_STEP_PIN         26
#define E_DIR_PIN          28
#define E_ENABLE_PIN       24

#define Q_STEP_PIN         36
#define Q_DIR_PIN          34
#define Q_ENABLE_PIN       30

#define SDPOWER            -1
#define SDSS               53
#define LED_PIN            13

#define FAN_PIN            9

#define PS_ON_PIN          12
#define KILL_PIN           -1

#define HEATER_0_PIN       10
#define HEATER_1_PIN       8
#define TEMP_0_PIN          13   // ANALOG NUMBERING
#define TEMP_1_PIN          14   // ANALOG NUMBERING


//Limit Switch Pins


//Setup Load Cell pins
#define DOUT  23
#define CLK   25
#define calibration_factor -7050.0 //*0.453592

HX711 scale(DOUT, CLK);

#define DOUT2  27
#define CLK2   29
#define calibration_factor2 -7050.0 //*0.453592

HX711 scale2(DOUT2, CLK2);

//SoftwareSerial smcSerial = SoftwareSerial(rxPin, txPin);

float currentX = 0;
float currentY = 0;
float currentZ = 0;
float revx;
float revy;
float revz;
int x;
int y;
int z;
int s;
float WOB1;
float WOB2;
float WOBavg;
int zmove;



// required to allow motors to move
//// must be called when controller restarts and after any error
//void exitSafeStart()
//{
//  smcSerial.write(0x83);
//}

//// speed should be a number from -3200 to 3200
//void setMotorSpeed(int speed)
//{
//  if (speed < 0)
//  {
//    smcSerial.write(0x86);  // motor reverse command
//    speed = -speed;  // make speed positive
//  }
//  else
//  {
//    smcSerial.write(0x85);  // motor forward command
//  }
//  smcSerial.write(speed & 0x1F);
//  smcSerial.write(speed >> 5);
//}


void setup() {
//  // initialize software serial object with baud rate of 19.2 kbps
////  smcSerial.begin(19200);
//  // the Simple Motor Controller must be running for at least 1 ms
//  delay(5);


//  smcSerial.write(0xAA);  // send baud-indicator byte

  // Sets the two pins as Outputs
  pinMode(Drill_Speed_Pin, OUTPUT);
  pinMode(FAN_PIN , OUTPUT);
  pinMode(HEATER_0_PIN , OUTPUT);
  pinMode(HEATER_1_PIN , OUTPUT);
  pinMode(LED_PIN  , OUTPUT);

  pinMode(X_STEP_PIN  , OUTPUT);
  pinMode(X_DIR_PIN    , OUTPUT);
  pinMode(X_ENABLE_PIN    , OUTPUT);
  pinMode(X_MIN_PIN, INPUT_PULLUP);
  pinMode(X_MAX_PIN, INPUT_PULLUP);

  pinMode(Y_STEP_PIN  , OUTPUT);
  pinMode(Y_DIR_PIN    , OUTPUT);
  pinMode(Y_ENABLE_PIN    , OUTPUT);
  pinMode(Y_MIN_PIN, INPUT_PULLUP);
  pinMode(Y_MAX_PIN, INPUT_PULLUP);

  pinMode(Z_STEP_PIN  , OUTPUT);
  pinMode(Z_DIR_PIN    , OUTPUT);
  pinMode(Z_ENABLE_PIN    , OUTPUT);
  pinMode(Z_MIN_PIN, INPUT_PULLUP);
  pinMode(Z_MAX_PIN, INPUT_PULLUP);


  pinMode(E_STEP_PIN  , OUTPUT);
  pinMode(E_DIR_PIN    , OUTPUT);
  pinMode(E_ENABLE_PIN    , OUTPUT);

  pinMode(Q_STEP_PIN  , OUTPUT);
  pinMode(Q_DIR_PIN    , OUTPUT);
  pinMode(Q_ENABLE_PIN    , OUTPUT);

  digitalWrite(X_ENABLE_PIN    , LOW);
  digitalWrite(Y_ENABLE_PIN    , LOW);
  digitalWrite(Z_ENABLE_PIN    , LOW);
  digitalWrite(E_ENABLE_PIN    , LOW);
  digitalWrite(Q_ENABLE_PIN    , LOW);
  analogWrite(Drill_Speed_Pin, 255);
  delay(1000);
  analogWrite(Drill_Speed_Pin, 120);
  delay(1000);
//  analogWrite(Drill_Speed_Pin, 0);
  digitalWrite(Drill_Speed_Pin, LOW);
  Serial.begin(9600);
  Serial.println("Start");

  //Serial2.begin(9600);

  //zero load cell 1
  scale.set_scale(calibration_factor);
  scale.tare();
  //zero load cell 2
  scale2.set_scale(calibration_factor2);
  scale2.tare();

//  exitSafeStart();  // clear the safe-start violation and let the motor run
//  // initialize i2c as slave
//  Wire.begin(slaveAddress);  // I2C slave address 8 setup.
//  // Wire.onReceive(i2cReceive);  // register our handler function with the Wire library
//  // Wire.onRequest(i2cTransmit);  // register data return handler
 HomePos();
//  zmove = 0;

}
void loop() {
  if (Serial.available() > 0) {
    int sercommand = Serial.read();
    switch (sercommand) {
      case '1':
        Serial.println("Enter number of mm for x(whole numbers only!)");
         while (Serial.available() == 0) { }
        x = Serial.parseFloat();

        Serial.println("Enter number of mm for y(whole numbers only!)");
        while (Serial.available() == 0) { }
        y = Serial.parseFloat();

        DrillXYPos(x, y);


        currentX = x;
        currentY = y;
        Serial.println("DONE!");
        break;
      //Z position and drill speed
      case '2': // z down
        digitalWrite(Z_DIR_PIN, LOW);
        if (zmove == 2) {
          zmove = 1;
          Serial.println(zmove);
          DrillZPos(zmove);
        }
        else {
          zmove = 1;
          Serial.println(zmove);
          DrillZPos(zmove);
        }
        break;
      case '3':
        digitalWrite(Z_DIR_PIN, HIGH);
        if (zmove == 1) {
          zmove = 2;
          Serial.println(zmove);
          DrillZPos(zmove);
        }
        else {
          zmove = 2;
          Serial.println(zmove);
          DrillZPos(zmove);
        }
        break;
      //  Serial.println("Input drill depth in mm(0 means return to top)");
      //  while (Serial.available()==0){ }
      //  int z = Serial.parseInt();
      //
      //  Serial.println("Input drill speed in from 1 to 3200");
      //  while (Serial.available()==0){ }
      //  int s = Serial.parseInt();
      //
      //  //Serial.print("fuck me in the ass");
      //
      //  DrillZPos(z, s);
      //
      //  setMotorSpeed(0);
      //  currentZ = z;
      //
      //  Serial.println("DONE!");
      //  delay(1000); // One second delay
      //}
      case '4':
//        setMotorSpeed(3200);
        break;
      case '5':
//        setMotorSpeed(0);
        break;
      case '6':
//        setMotorSpeed(-3200);
        break;
    }
  }
}
//void i2cReceive(int bytes){
//  int command = Wire.read(); //should this be byte?
//
//  switch(command){
//    case 0x0A: //position drill in x and y
//    x = Wire.read();
//    y = Wire.read();
//    if (digitalRead(Z_MIN_PIN) == HIGH){
//    DrillXYPos(x,y);}
//    else{
//      digitalWrite(Z_DIR_PIN, LOW);
//      while(digitalRead(Z_MIN_PIN) == LOW){
//        digitalWrite(Z_STEP_PIN, HIGH);
//        delayMicroseconds(500);
//        digitalWrite(Z_STEP_PIN, LOW);
//        delayMicroseconds(500);
//      }
//      DrillXYPos(x,y);}
//    break;
//
//    case 0x0B: //drill z positioning
//    z = Wire.read();
//    s = Wire.read();
//    Serial.println(z);
//    Serial.println(s);
//    s = s * 32;
//    DrillZPos(z,s);
//    break;
//
//    case 0x0C: //send home
//    //HomePos();
//    break;
//
//    default:
//    //do nothing
//    break;
//
//  }
//
//
//
//
//}
//
//void i2cTransmit() {
//  WOB1=scale.get_units();
//  WOB2=scale2.get_units();
//  WOBavg=(WOB1+WOB2)/2;
//  wireWriteData(WOBavg);
//
//
//
//
//  // Both the Arduino and RPi are little-endian, no conversion needed...
//  //Wire.write((uint8_t *)&sensorCurr, sizeof(sensorCurr));
//}
