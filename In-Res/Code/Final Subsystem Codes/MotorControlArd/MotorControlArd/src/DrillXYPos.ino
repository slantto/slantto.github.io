#include <Arduino.h>

void DrillXYPos(float x,float y)
{
  if (x < currentX) {
        digitalWrite(xdirPin,LOW);
        float mmx = currentX-x;
        float revx = 65.625 * mmx;
        Serial.println("move x");
        for (int j = 0; j < revx; j++) {
      digitalWrite(xstepPin,HIGH); 
      delayMicroseconds(500); 
      digitalWrite(xstepPin,LOW); 
      delayMicroseconds(500); 
    }
    }
    
    else if (x > currentX){
        digitalWrite(xdirPin,HIGH);
        float mmx = x- currentX;
        float revx = 65.625 * mmx;
        Serial.println("move x");
        for (int j = 0; j < revx; j++) {
      digitalWrite(xstepPin,HIGH); 
      delayMicroseconds(500); 
      digitalWrite(xstepPin,LOW); 
      delayMicroseconds(500); 
  }
    }
    
    else if (x == currentX){
      Serial.println("Not move x");
    }
    
    if (y < currentY) {
        digitalWrite(ydirPin,LOW);
        float mmy = currentY - y;
        float revy = 65.625 * mmy;
        Serial.println("move y");
        for (int j = 0; j < revy; j++) {
      digitalWrite(ystepPin,HIGH); 
      delayMicroseconds(500); 
      digitalWrite(ystepPin,LOW); 
      delayMicroseconds(500); 
  }
    }
    
    else if (y > currentY){
        digitalWrite(ydirPin,HIGH);
        float mmy = y-currentY;
        float revy = 65.625 * mmy;
        Serial.println("move y");
        for (int j = 0; j < revy; j++) {
      digitalWrite(ystepPin,HIGH); 
      delayMicroseconds(500); 
      digitalWrite(ystepPin,LOW); 
      delayMicroseconds(500); 
  }
    }
    
    else if (y == currentY){
      Serial.println("not move y");
      }
      
}
    
