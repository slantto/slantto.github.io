void DRILL_OP(float X,float Z, float meltime){
  
  DrillXZPos(X, Z);
  
  scale2.set_scale();
  scale2.tare();	//Reset the scale2 to 0

  
  digitalWrite(Y_DIR_PIN,HIGH);
  digitalWrite(DRILL_DIR_PIN, LOW);
  digitalWrite(DRILL_SPEED_PIN, HIGH);
  //unsigned long tstart = 0;

  while(digitalRead(Y_MAX_PIN) == HIGH){
    if ((WOBavg < WOBmax) && (WOBavg < WOBthresh)){
      deladj = 500;
      
      if (WOBavg >= WOBthresh){
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
    else if(WOBavg >= WOBmax){
      digitalWrite(Y_DIR_PIN,LOW);
      unsigned long deltime = millis();
      while((millis()-deltime)<2000){
        digitalWrite(Y_STEP_PIN, HIGH); 
      delayMicroseconds(deladj); 
      digitalWrite(Y_STEP_PIN, LOW); 
      delayMicroseconds(deladj); }
      digitalWrite(Y_DIR_PIN,HIGH);
      //digitalWrite(DRILL_DIR_PIN,LOW);
      //delay(2500);
    }
    
    if((millis()-tstart) > 5000){
      tstart = millis();
      readvalue = analogRead(CSENSE_PIN);
     Voltage = (readvalue*5)/1024;
VRMS = (Voltage/2.0) * 0.707;
IRMS = (VRMS *1000)/mVperAmp;
     Serial.print("<Arms= ");
     Serial.print(IRMS);
     Serial.println(" >");
     //WOB1=scale.get_units();
    // scale2.tare();
      scale2.set_scale(calibration_factor2); //Adjust to this calibration factor
     WOB2=scale2.get_units();
     //WOBavg=(WOB1+WOB2)/2;
     WOBavg = WOB2 * 4;
   // Serial.print("<W1=");
    //Serial.print(abs(WOB1));
    Serial.print("<WOB = ");
    //Serial.print(abs(WOB2));
    //Serial.print(", WA=");
    Serial.print(abs(WOBavg));
    Serial.println(" >");
        }
        if(Serial.available())
  {
    char temp = Serial.read();
    if(temp == 'a')
      digitalWrite(Y_DIR_PIN,LOW);
    else if(temp == 'z')
      digitalWrite(Y_DIR_PIN,HIGH);
  }  
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

