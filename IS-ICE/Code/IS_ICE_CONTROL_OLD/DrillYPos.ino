void DrillYPos(float Y,int s)
{
   if ( Y <  currentY) {
        digitalWrite(Y_DIR_PIN,HIGH);
        float mmY = currentY-Y;
        float stepY = 78.74 * 4 * mmY; //78.74 steps per mm atfull step
        int j = 0;
        while ( j < stepY ) {  
            digitalWrite(Y_STEP_PIN,HIGH);
            delayMicroseconds(500);
            digitalWrite(Y_STEP_PIN,LOW);
            delayMicroseconds(500);
            j++;
            if (digitalRead(Y_MIN_PIN)==LOW){
              Serial.println("YMIN is Pressed!");
              Y = 0;
              currentY = 0;
              break;}
        }
  }
  
    else if (Y > currentY){
        digitalWrite(Y_DIR_PIN,LOW);
        float mmY = Y - currentY;
        float stepY = 78.74 * 4 * mmY;//78.74 steps per mm for fullstep
        int j = 0;
        digitalWrite(DRILL_DIR_PIN, HIGH);
        analogWrite(DRILL_SPEED_PIN, s);
        while (j<stepY){
          if ( (WOBavg > -20)&&(digitalRead(Y_MAX_PIN) == HIGH)){
            digitalWrite(Y_STEP_PIN,HIGH);
            delayMicroseconds(500);
            digitalWrite(Y_STEP_PIN,LOW);
            delayMicroseconds(500);
            j++;
            if (digitalRead(Y_MAX_PIN)==LOW){
              Serial.println("YMAX is Pressed!");
              j = stepY;
              break;}
          }
          else {}
            
          
           WOB1=scale.get_units();
           WOB2=scale2.get_units();
           WOBavg=(WOB1+WOB2)/2;
           if(WOBavg < -3){
           Serial.write((byte)abs(WOB1));
           Serial.write((byte)abs(WOB2));
           Serial.write((byte)abs(WOBavg));}
    }

    }
    else if (Y == currentY){
      digitalWrite(DRILL_DIR_PIN, HIGH);
      analogWrite(DRILL_SPEED_PIN, s);
      Serial.println("on for 10 seconds");
      delay(10000);

    }

  currentY = Y;
  Serial.println(currentY);
  return;
  
}
