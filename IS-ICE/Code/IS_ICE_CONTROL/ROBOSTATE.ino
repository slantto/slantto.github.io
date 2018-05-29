void ROBOSTATE(){

//FIND Irms
double Irms = emon1.calcIrms(1480);  // Calculate Irms only
Serial.print(Irms*120.0);           // Apparent power
  Serial.print(" ");
  Serial.println(Irms);             // Irms
  
Serial.println(currentX);
Serial.print(" ");
Serial.print(currentY);
Serial.print(" ");
Serial.print(currentZ);


WOB1=scale.get_units();
WOB2=scale2.get_units();
WOBavg=(WOB1+WOB2)/2;

Serial.println(WOB1);
Serial.print(" ");
Serial.print(WOB2);
Serial.print(" ");
Serial.print(WOBavg);
     
}
