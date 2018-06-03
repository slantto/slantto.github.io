void replyToPi(){
  
  if(newDataFromPC){
    newDataFromPC = false;
    Serial.print("<Command ");
    Serial.print(messageFromPC);
    Serial.print("SwitchCom");
    Serial.print(switchCom);
    Serial.print(", COMPARAM1 = ");
    Serial.print(COMPARAM1);
    Serial.print(", COMPARAM2 = ");
    Serial.print(COMPARAM2);
    Serial.print(", COMPARAM3 = ");
    Serial.print(COMPARAM3);
    Serial.println(">");
    commandReady = true;
  }
  
}
