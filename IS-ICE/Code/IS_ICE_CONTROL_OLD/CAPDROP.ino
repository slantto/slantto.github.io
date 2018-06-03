void CAPDROP(){
  
  capdrop = 90;
capservo.write(capdrop);
delay(500);
capdrop = 0;
capservo.write(capdrop);
delay(500);

}