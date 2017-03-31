#include <Arduino.h>

void DrillZPos(float z,int s)
{
   if ( z <  currentZ) {
        digitalWrite(zdirPin,LOW);
        float mmz = currentZ-z;
        float revz = 65.625 * mmz;
        setMotorSpeed(s*-1);
        Serial.println("machine on");
        for(int j = 0; j < revz; j++) {
      digitalWrite(zstepPin,HIGH); 
      delayMicroseconds(500); 
      digitalWrite(zstepPin,LOW); 
      delayMicroseconds(500); 
  }
    }
    else if (z > currentZ){
        digitalWrite(zdirPin,HIGH);
        float mmz = z - currentZ;
        float revz = 65.625 * mmz;
        setMotorSpeed(s);
        Serial.println("machine on");
        for(int j = 0; j < revz; j++) {
      digitalWrite(zstepPin,HIGH); 
      delayMicroseconds(500); 
      digitalWrite(zstepPin,LOW); 
      delayMicroseconds(500); 
  }
    }
    else if (z == currentZ){
      setMotorSpeed(s);
      Serial.println("on");
      delay(10000);
      
    }
   
  setMotorSpeed(0);
  currentZ = z;  
}
 
