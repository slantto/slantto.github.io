#include <Arduino.h>

#include <Wire.h>

void setup() {
  Wire.begin();
  Serial.begin(9600);
}

void loop() {
  Wire.requestFrom(4,6);

  while (Wire.available()){
    char c = Wire.read();
    Serial.print(c);
  }
delay(500);

 Wire.requestFrom(6,6);

  while (Wire.available()){
    char b = Wire.read();
    Serial.print(b);
  }
delay(500);
   Wire.requestFrom(8,6);

  while (Wire.available()){
    char d = Wire.read();
    Serial.print(d);
  }
  delay(500);
}
