
//This is the Motor Control Arduino Code it takes input from the raspberry pi and performs the selected command. 

//include libraries
#include <HX711.h>
#include <math.h>
#include <Servo.h>

//Drill motor pin
#define DRILL_SPEED_PIN    A11 // pin A11(D65)
#define DRILL_DIR_PIN 59

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
#define CSENSE_PIN         A15

#define PS_ON_PIN          12
#define KILL_PIN           -1

#define HEATER_0_PIN       10
#define HEATER_1_PIN       8
#define TEMP_0_PIN         A13   // ANALOG NUMBERING
#define TEMP_1_PIN         A14   // ANALOG NUMBERING

//Setup Load Cell pins
#define DOUT  40
#define CLK   63
#define calibration_factor 500 //*0.453592
HX711 scale(DOUT, CLK);

#define DOUT2  57
#define CLK2   58
#define calibration_factor2 500 //*0.453592
HX711 scale2(DOUT2,CLK2);

#define THERM_R_NOM 100000
#define TEMP_NOM 25
#define BETAVAL 3950
#define PULLR 4700

float SETTEMP = 50;
float TEMP_MAX = 100;

//Create a servo object
Servo capservo;

// Parameters for measuring RMS current
const double vRMS = 120.0;     // RMS voltage
const double offset = 2.5;     // Half the ADC max voltage
const int numTurns = 2000;      // 1:2000 transformer turns
const int rBurden = 100;        // Burden resistor value in Ohms
const int numSamples = 1000;    // Number of samples before calculating RMS


float currentX=0;
float currentY=0;
float currentZ=0;

float revX;
float revY;
float revZ;

int X;
int Y;
int Z;
int s;
int meltime;

float WOB1;
float WOB2;
float WOBavg;
float WOBthresh = -15;
float WOBmax = -20;
float deladj;

int sample;
double voltage;
double iPrimary;
double acc = 0;
double iRMS;
double apparentPower;

int capdrop = 0;

void setup() {

  // Sets the two pins as Outputs
  pinMode(DRILL_SPEED_PIN, OUTPUT);
  pinMode(DRILL_DIR_PIN, OUTPUT);
  pinMode(FAN_PIN , OUTPUT);
  pinMode(HEATER_0_PIN , OUTPUT);
  pinMode(HEATER_1_PIN , OUTPUT);
  pinMode(LED_PIN  , OUTPUT);
  pinMode(TEMP_0_PIN, INPUT);
  pinMode(TEMP_1_PIN, INPUT);
  pinMode(CSENSE_PIN, INPUT);
  
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
 
 digitalWrite(HEATER_1_PIN, LOW);
 digitalWrite(DRILL_DIR_PIN, LOW);
 analogWrite(DRILL_SPEED_PIN, 255);
 delay(3000);
analogWrite(DRILL_SPEED_PIN, 0);
delay(500);
 digitalWrite(DRILL_DIR_PIN, HIGH);
 analogWrite(DRILL_SPEED_PIN, 120);
 delay(3000);
//  analogWrite(DRILL_SPEED_PIN, 0);
 digitalWrite(DRILL_SPEED_PIN, LOW);
 
 Serial.begin(9600);
  // Serial2.begin(9600);

//attach servo object to pin D11 
capservo.attach(11);

capdrop = 90;
capservo.write(capdrop);
delay(500);
capdrop = 0;
capservo.write(capdrop);
delay(500);

//Zero load cell 1
scale.set_scale(calibration_factor);
scale.tare();
//Zero load cell 2
scale2.set_scale(calibration_factor2);
scale2.tare();

  HomePos();
  Serial.println("Ready");

}
void loop() {
  if (Serial.available() > 0) {
    int sercommand = Serial.read();
    switch (sercommand) {
      case '1': //Position drill in X and Z
        Serial.println("Enter number of mm for X(whole numbers only!)");
         while (Serial.available() == 0) { }
        X = Serial.parseFloat();

        Serial.println("Enter number of mm for Z(whole numbers only!)");
        while (Serial.available() == 0) { }
        Z = Serial.parseFloat();

        DrillXZPos(X, Z);


        currentX = X;
        currentZ = Z;
        Serial.println("X = ");
        Serial.print(currentX);
        Serial.println("Z = ");
        Serial.print(currentZ);
        Serial.println("Command Executed!");
        delay(1000);
        break;
      //Z position and drill speed
      case '2': //position drill in vertical
          Serial.println("Input drill depth in mm(0 means return to top)");
        while (Serial.available()==0){ }
        Y = Serial.parseInt();
      
        Serial.println("Input drill speed in from 1 to 3200");
        while (Serial.available()==0){ }
        s = Serial.parseInt();
      
        DrillYPos(Y, s);
      
        currentY = Y;
      
        Serial.println("Y = ");
        Serial.print(currentY);
        Serial.println("Command Executed!");
        delay(1000); // One second delay
        break;
      case '3': //Drill operation
        Serial.println("Enter number of mm for X(whole numbers only!)");
         while (Serial.available() == 0) { }
        X = Serial.parseFloat();

        Serial.println("Enter number of mm for Z(whole numbers only!)");
        while (Serial.available() == 0) { }
        Z = Serial.parseFloat();
        
        Serial.println("Enter minutes for melt chamber to melt ice(WHOLE NUMBERS ONLY, I WILL FIND YOU)");
        while (Serial.available() == 0) { }
        meltime = Serial.parseFloat();
        
        DRILL_OP(X, Z, meltime);
        
        Serial.println("Drill Op completed");
        break;
      case '4': //turn heater on and off
      if(digitalRead(HEATER_0_PIN)==LOW){
        digitalWrite(HEATER_0_PIN, HIGH);
        Serial.println("Heater ON!");
      }
      else if(digitalRead(HEATER_0_PIN)==HIGH){
        digitalWrite(HEATER_0_PIN, LOW);
        Serial.println("Heater OFF!");}
        
        
        break;
      case '5': //Robot State
        ROBOSTATE();
        break;
      case '6':
      
        break;
    }
  }
}
