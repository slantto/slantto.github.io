void MELT_CHAMBER(float meltime){
  
 meltime = meltime * 60 * 1000; //convert from minutes to millis
 const long interval = meltime;
 unsigned long previousMillis = 0;

 unsigned long currentMillis = millis();

while (currentMillis-previousMillis < interval){
  unsigned long currentMillis = millis();
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
  
  if (TEMP_C > SETTEMP+2){
    digitalWrite(HEATER_1_PIN,LOW);
  }
  
  else if (TEMP_C < SETTEMP-2){
    digitalWrite(HEATER_1_PIN,HIGH);
  }
  
  if (TEMP_C>= TEMP_MAX){
    digitalWrite(HEATER_1_PIN,LOW);
    Serial.println("CAUTION! MELT CHAMBER EXCEEDING MAXIMUM TEMPERATURE!");
  }
  
  } 

  digitalWrite(HEATER_1_PIN,LOW);
  
}