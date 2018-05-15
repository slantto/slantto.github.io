void HomePos(){
  
//        //bring z home
//      if(digitalRead(Z_MIN_PIN) == HIGH){
//        digitalWrite(Z_DIR_PIN, HIGH);
//      }
//      while(digitalRead(Z_MIN_PIN) == HIGH){
//        digitalWrite(Z_STEP_PIN, HIGH); 
//        delayMicroseconds(500); 
//        digitalWrite(Z_STEP_PIN, LOW); 
//        delayMicroseconds(500); 
//      }
//bring X home
      if(digitalRead(X_MIN_PIN)== LOW){
        digitalWrite(X_DIR_PIN, LOW);
      }
      while(digitalRead(X_MIN_PIN) == LOW){
        digitalWrite(X_STEP_PIN,HIGH); 
        delayMicroseconds(500); 
        digitalWrite(X_STEP_PIN,LOW); 
        delayMicroseconds(500); 
      }

// bring y home
      if(digitalRead(Y_MIN_PIN) == HIGH){
        digitalWrite(Y_DIR_PIN, HIGH);
      }
      while(digitalRead(Y_MIN_PIN) == HIGH){
        digitalWrite(Y_STEP_PIN, HIGH); 
        delayMicroseconds(500); 
        digitalWrite(Y_STEP_PIN,LOW); 
        delayMicroseconds(500);  
      }


  currentX = 0;
  currentY = 0;
  currentZ = 0;
  x = 0;
  y = 0;
  z = 0;
}
