#include <Arduino.h>

// defines pins numbers
const int xstepPin = 3; 
const int xdirPin = 4;
const int ystepPin = 5; 
const int ydirPin = 6;
const int zstepPin = 7; 
const int zdirPin = 8;
 
void setup() {
  // Sets the two pins as Outputs
  pinMode(xstepPin,OUTPUT); 
  pinMode(xdirPin,OUTPUT);
  pinMode(ystepPin, OUTPUT);
  pinMode(ydirPin,OUTPUT);
  pinMode(zstepPin, OUTPUT);
  pinMode(zdirPin,OUTPUT);
  Serial.begin(9600);
  
}
void loop() {
  Serial.println("select a motor 1=x,2=y,3=z");
  while (Serial.available()==0){ }
  int moto = Serial.parseInt();
  
  if (moto == 1){
  Serial.println("what direction 1 for CW, 2 for CCW");
  while (Serial.available()==0){ }
  int dir = Serial.parseInt();
  if (dir == 1){
    digitalWrite(xdirPin,HIGH);
     }
    else if (dir == 2){
        digitalWrite(xdirPin,LOW);
    }
  Serial.println("Enter number of mm(whole numbers only!)");
  while (Serial.available()==0){ }
  float mm = Serial.parseFloat();
  float rev = 62.5 * mm ;
  // 20000 steps per revolution 0.018 degree per step
  for(int x = 0; x < rev; x++) {
    digitalWrite(xstepPin,HIGH); 
    delayMicroseconds(500); 
    digitalWrite(xstepPin,LOW); 
    delayMicroseconds(500); 
  }
  }
  
else if (moto == 2){
  Serial.println("what direction 1 for CW, 2 for CCW");
  while (Serial.available()==0){ }
  int dir = Serial.parseInt();
  if (dir == 1){
    digitalWrite(ydirPin,HIGH);
     }
    else if (dir == 2){
        digitalWrite(ydirPin,LOW);
    }
  Serial.println("Enter number of mm(whole numbers only!)");
  while (Serial.available()==0){ }
  float mm = Serial.parseFloat();
  float rev = 62.625 * mm ;
  // 20000 steps per revolution 0.018 degree per step
  for(int x = 0; x < rev; x++) {
    digitalWrite(ystepPin,HIGH); 
    delayMicroseconds(500); 
    digitalWrite(ystepPin,LOW); 
    delayMicroseconds(500); 
  }
}
  
   else if (moto == 3){
  Serial.println("what direction 1 for CW, 2 for CCW");
  while (Serial.available()==0){ }
  int dir = Serial.parseInt();
  if (dir == 1){
    digitalWrite(zdirPin,HIGH);
     }
    else if (dir == 2){
        digitalWrite(zdirPin,LOW);
    }
  Serial.println("Enter number of mm(whole numbers only!)");
  while (Serial.available()==0){ }
  float mm =Serial.parseFloat();
  float rev = 262.5 * mm ;
  // 20000 steps per revolution 0.018 degree per step
  for(int x = 0; x < rev; x++) {
    digitalWrite(zstepPin,HIGH); 
    delayMicroseconds(500); 
    digitalWrite(zstepPin,LOW); 
    delayMicroseconds(500); 
  }
  
  }
  Serial.println("DONE!");
  delay(1000); // One second delay
  
  
}
  
