void ROBOSTATE(){


Serial.print("<");

Voltage = getVPP();
VRMS = (Voltage/2.0) * 0.707;
IRMS = (VRMS *1000)/mVperAmp;


 float samples;
  int i;
  float average;
  
  for (i=0; i<5; i++){
    samples += analogRead(TEMP_0_PIN);
    delay(10);
  }
  
  average = samples/5;
  
  average = 1023/average -1;
  average = PULLR/average;
  
  float TEMP_C;
  TEMP_C = average/THERM_R_NOM;
  TEMP_C = log(TEMP_C);
  TEMP_C /= BETAVAL;
  TEMP_C += 1.0/(TEMP_NOM + 273.15);
  TEMP_C = 1/TEMP_C;
  TEMP_C -= 273.15;
  
  Serial.print("<");
  Serial.print(IRMS);
Serial.print(" ");
Serial.print(currentX);
Serial.print(" ");
Serial.print(currentY);
Serial.print(" ");
Serial.print(currentZ);
  Serial.print(TEMP_C);
  Serial.println(" Celsius>");
  //Serial.println("done");

}
