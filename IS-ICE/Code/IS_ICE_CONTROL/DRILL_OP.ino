void DRILL_OP(float X,float Z, float meltime)
{
  float readvalue;
  DrillXZPos(X, Z);
  
  
  digitalWrite(Y_DIR_PIN, LOW);
  digitalWrite(DRILL_DIR_PIN, HIGH);
  digitalWrite(DRILL_SPEED_PIN, HIGH);
  
  
  while(digitalRead(Y_MAX_PIN) == HIGH){
    if ((WOBavg > WOBmax) && (WOBavg > WOBthresh)){
      deladj = 500;
      if (WOBavg <= WOBthresh){
        deladj = 500 + (100*(abs(WOBthresh) - abs(WOBavg)));
      }
      digitalWrite(Y_STEP_PIN, HIGH); 
      delayMicroseconds(deladj); 
      digitalWrite(Y_STEP_PIN, LOW); 
      delayMicroseconds(deladj); 
    }
    // else if((WOBavg > WOBmax) && (WOBavg <= WOBthresh)){
    //   deladj = 500 + (100*(abs(WOBthresh) - abs(WOBavg)));
    //   digitalWrite(Y_STEP_PIN, HIGH); 
    //   delayMicroseconds(500); 
    //   digitalWrite(Y_STEP_PIN, LOW); 
    //   delayMicroseconds(500); 
    // }
    else if(WOBavg <= WOBmax){
      delay(2000);
    }
     readvalue = (((analogRead(CSENSE_PIN)*5.0)/1024.0)*1000)/mVperAmp;
     Serial.print("<Amps=");
     Serial.print(readvalue);
     Serial.println(">");
     WOB1=scale.get_units();
     WOB2=scale2.get_units();
     WOBavg=(WOB1+WOB2)/2;
     WOBavg = WOB2 * 4;
    // if(WOBavg < -5){
    Serial.print("<W1=");
    Serial.print(abs(WOB1));
    Serial.print(", W2=");
    Serial.print(abs(WOB2));
    Serial.print(", WA=");
    Serial.print(abs(WOBavg));
    Serial.println(">");
    }
    
    
  digitalWrite(DRILL_SPEED_PIN, LOW);
  
  HomePos();
  
  digitalWrite(DRILL_DIR_PIN, LOW);
  analogWrite(DRILL_SPEED_PIN, 128);
  delay(15000);
  digitalWrite(DRILL_SPEED_PIN, LOW);
  
  MELT_CHAMBER(meltime);
  
  currentX = 0;
  currentY = 0;
  currentZ = 0;
  X = 0;
  Y = 0;
  Z = 0;
  
  //Serial.println("Drill Op Complete ");
  
}
