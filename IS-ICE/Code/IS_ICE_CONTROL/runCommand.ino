void runCommand(){
Serial.print("<begin");
Serial.print(messageFromPC);
//Serial.print(">");
//int switchCom = str(messageFromPC).toInt();
Serial.print("execute");
Serial.print(switchCom);
Serial.println("end>");
switch(switchCom) {
   case 1: //Position drill in X and Z
      X = COMPARAM1;
      Z = COMPARAM2;
      //Serial.println("<zexecute>");
      DrillXZPos(X, Z);

      currentX = X;
      currentZ = Z;
      Serial.print("<X = ");
      Serial.print(currentX);
      Serial.print("Z = ");
      Serial.print(currentZ);
      Serial.println(">");
      Serial.println("<done>");
      delay(1000);
      break;

    case 2: //position drill in vertical
      Y = COMPARAM1;
      s = COMPARAM2;
      
      DrillYPos(Y, s);
      
      currentY = Y;
      
      Serial.print("<Y = ");
      Serial.print(currentY);
      Serial.println(">");
      Serial.print("<done>");
      delay(1000); // One second delay
      break;
      
    case 3: //Drill operation
      X = COMPARAM1;
      Z = COMPARAM2;
      meltime = COMPARAM3;
        
      DRILL_OP(X, Z, meltime);
        
      Serial.println("<Drill Operation Complete>");
      break;
      
    case 4: //turn heater on and off
    
      if(digitalRead(HEATER_0_PIN)==LOW){
        digitalWrite(HEATER_0_PIN, HIGH);
        Serial.println("<Heater ON!>");
      }
      else if(digitalRead(HEATER_0_PIN)==HIGH){
        digitalWrite(HEATER_0_PIN, LOW);
        Serial.println("<Heater OFF!>");}
        
      Serial.print("<done>");
      break;
        
    case 5: //Robot State
      
      ROBOSTATE();
      Serial.print("<done>");
      break;
    
    case 6:
    
      Serial.println("<Execute Order 66>");
      
      break;
}
  
}

