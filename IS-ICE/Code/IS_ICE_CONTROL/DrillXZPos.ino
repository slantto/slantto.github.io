void DrillXZPos(float X,float Z){
  
  
  if (X < currentX) {
        digitalWrite(X_DIR_PIN,LOW);
        float mmX = currentX-X;
        float revX = 65.625 * mmX;
        //Serial.println("move X");
        for (int j = 0; j < revX; j++) {
      digitalWrite(X_STEP_PIN,HIGH); 
      delayMicroseconds(500); 
      digitalWrite(X_STEP_PIN,LOW); 
      delayMicroseconds(500); 
      if (digitalRead(X_MIN_PIN)==LOW){
        Serial.println("<XMIN is Pressed!>");
        currentX = 0;
        X = 0;
        break;}
    }
    }
    
    else if (X > currentX){
        digitalWrite(X_DIR_PIN,HIGH);
        float mmX = X- currentX;
        float revX = 65.625 * mmX;
        //Serial.println("move X");
        for (int j = 0; j < revX; j++) {
      digitalWrite(X_STEP_PIN,HIGH); 
      delayMicroseconds(500); 
      digitalWrite(X_STEP_PIN,LOW); 
      delayMicroseconds(500); 
      if (digitalRead(X_MAX_PIN)==HIGH){
        Serial.println("<XMAX is Pressed!>");
        break;}
  }
    }
    
    else if (X == currentX){
      //Serial.println("Not move X");
    }
    
    if (Z < currentZ) {
        digitalWrite(Z_DIR_PIN,LOW);
        float mmZ = currentZ - Z;
        float revZ = 65.625 * mmZ;
        //Serial.println("move Z");
        for (int j = 0; j < revZ; j++) {
      digitalWrite(Z_STEP_PIN,HIGH); 
      delayMicroseconds(500); 
      digitalWrite(Z_STEP_PIN,LOW); 
      delayMicroseconds(500); 
      if (digitalRead(Z_MIN_PIN)==LOW){
        Serial.println("<ZMIN is Pressed!>");
        currentZ = 0;
        Z = 0;
        break;}
  }
    }
    
    else if (Z > currentZ){
        digitalWrite(Z_DIR_PIN,HIGH);
        float mmZ = Z-currentZ;
        float revZ = 65.625 * mmZ;
        //Serial.println("move Z");
        for (int j = 0; j < revZ; j++) {
      digitalWrite(Z_STEP_PIN,HIGH); 
      delayMicroseconds(500); 
      digitalWrite(Z_STEP_PIN,LOW); 
      delayMicroseconds(500); 
      if (digitalRead(Z_MAX_PIN)==LOW){
        Serial.println("<ZMAX is Pressed!>");
        break;}
  }
    }
    
    else if (Z == currentZ){
      //Serial.println("not move Z");
      }
      
      return;
}
    
