void DrillXYPos(float x,float y)
{
  
  
  if (x < currentX) {
        digitalWrite(X_DIR_PIN,LOW);
        float mmx = currentX-x;
        float revx = 65.625 * mmx;
        Serial.println("move x");
        for (int j = 0; j < revx; j++) {
      digitalWrite(X_STEP_PIN,HIGH); 
      delayMicroseconds(500); 
      digitalWrite(X_STEP_PIN,LOW); 
      delayMicroseconds(500); 
      if (digitalRead(X_MIN_PIN)==LOW){
        Serial.println("XMIN is Pressed! ");
        currentX = 0;
        x = 0;
        break;}
    }
    }
    
    else if (x > currentX){
        digitalWrite(X_DIR_PIN,HIGH);
        float mmx = x- currentX;
        float revx = 65.625 * mmx;
        Serial.println("move x");
        for (int j = 0; j < revx; j++) {
      digitalWrite(X_STEP_PIN,HIGH); 
      delayMicroseconds(500); 
      digitalWrite(X_STEP_PIN,LOW); 
      delayMicroseconds(500); 
      if (digitalRead(X_MAX_PIN)==LOW){
        Serial.println("XMAX is Pressed! ");
        break;}
  }
    }
    
    else if (x == currentX){
      Serial.println("Not move x");
    }
    
    if (y < currentY) {
        digitalWrite(Y_DIR_PIN,HIGH);
        float mmy = currentY - y;
        float revy = 65.625 * mmy;
        Serial.println("move y");
        for (int j = 0; j < revy; j++) {
      digitalWrite(Y_STEP_PIN,HIGH); 
      delayMicroseconds(500); 
      digitalWrite(Y_STEP_PIN,LOW); 
      delayMicroseconds(500); 
      if (digitalRead(Y_MIN_PIN)==LOW){
        Serial.println("YMIN is Pressed! ");
        currentY = 0;
        y = 0;
        break;}
  }
    }
    
    else if (y > currentY){
        digitalWrite(Y_DIR_PIN,LOW);
        float mmy = y-currentY;
        float revy = 65.625 * mmy;
        Serial.println("move y");
        for (int j = 0; j < revy; j++) {
      digitalWrite(Y_STEP_PIN,HIGH); 
      delayMicroseconds(500); 
      digitalWrite(Y_STEP_PIN,LOW); 
      delayMicroseconds(500); 
      if (digitalRead(Y_MAX_PIN)==LOW){
        Serial.println("YMAX is Pressed! ");
        break;}
  }
    }
    
    else if (y == currentY){
      Serial.println("not move y");
      }
      
      return;
}
    
