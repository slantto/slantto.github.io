void DrillZPos(int zmove)
{
//   if ( z <  currentZ) {
//        digitalWrite(Z_DIR_PIN,HIGH);
//        float mmz = currentZ-z;
//        float stepz = 394 * 1 * mmz; //393.7 steps per mm for full step
//        setMotorSpeed(s);
//        //Serial.println("machine on");
//        int j = 0;
//        for (int j = 0; j < stepz; j++) {  
//            digitalWrite(Z_STEP_PIN,HIGH);
//            delayMicroseconds(500);
//            digitalWrite(Z_STEP_PIN,LOW);
//            delayMicroseconds(500);
//            //j++;
//            if (digitalRead(Z_MIN_PIN)==LOW){
//              Serial.println("ZMIN is Pressed!");
//              z = 0;
//              currentZ = 0;
//              break;}
//        }
//  }
//  
//    else if (z > currentZ){
//         digitalWrite(Z_DIR_PIN,LOW);
//        // digitalWrite(zMS1Pin,HIGH);
//        // digitalWrite(zMS2Pin, HIGH);
//        // digitalWrite(zMS2Pin, HIGH);
//        float mmz = z - currentZ;
//        float stepz = 394 * 1 * mmz;//393.7 steps per mm for fullstep
//        setMotorSpeed(s);
//        //Serial.println("machine on");
  //Serial.print("Fuck my asshole");
//        int j = 0;
        //while (zmove == 1){
        float stepz =394 *50;
        for (int j = 0; j < stepz; j++){
          //if ( (WOBavg > -10.25)&&(digitalRead(Z_MAX_PIN) == HIGH)){
            digitalWrite(Z_STEP_PIN,HIGH);
            delayMicroseconds(500);
            digitalWrite(Z_STEP_PIN,LOW);
            delayMicroseconds(500);
            //j++;
           // Serial.println(j);
            if (digitalRead(Z_MAX_PIN)==LOW){
              Serial.println("ZMAX is Pressed!");
              //j = stepz;
              break;}
            if (zmove == 2){
               if (digitalRead(Z_MIN_PIN)==LOW){
                Serial.println("ZMIN is Pressed!");
                break;} 
            }

          }
         // else {}
            
          
//           WOB1=scale.get_units();
//           WOB2=scale2.get_units();
//           WOBavg=(WOB1+WOB2)/2;
//           //if(WOBavg < -5){
//           Serial.println(WOB1);
//           Serial.println(WOB2);
//           Serial2.write(abs(WOB1));
//           Serial2.write(abs(WOB2));
//           Serial2.write(abs(WOBavg));//}
  return;
    }
// digitalWrite(zMS1Pin,LOW);
// digitalWrite(zMS2Pin,LOW);
// digitalWrite(zMS3Pin,LOW);
//    else if (z == currentZ){
//      setMotorSpeed(s);
//      Serial.println("on");
//      delay(10000);
//
//    }
//
//  setMotorSpeed(0);
//  currentZ = z;
//  Serial.println(currentZ);

  

