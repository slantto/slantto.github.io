void DRILL_OP(float X,float Z, float meltime)
{
  
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
    
     WOB1=scale.get_units();
     WOB2=scale2.get_units();
     WOBavg=(WOB1+WOB2)/2;
     if(WOBavg < -5){
     Serial.write((byte)abs(WOB1));
     Serial.write((byte)abs(WOB2));
     Serial.write((byte)abs(WOBavg));}
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
  
  Serial.println("Drill Op Complete ");
  
}