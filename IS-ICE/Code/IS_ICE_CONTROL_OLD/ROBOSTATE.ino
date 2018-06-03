void ROBOSTATE(){

// Read ADC, convert to voltage, remove offset
sample = analogRead(A0);
voltage = (sample * 3.3) / 4096;
voltage = voltage - offset;
// Calculate the sensed current
iPrimary = (voltage / rBurden) * numTurns;

Serial.println(iPrimary);

Serial.println(currentX);
Serial.print(" ");
Serial.print(currentY);
Serial.print(" ");
Serial.print(currentZ);

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
  
  Serial.println(TEMP_C);
  Serial.print(" Celsius");
  Serial.println("done");

}
