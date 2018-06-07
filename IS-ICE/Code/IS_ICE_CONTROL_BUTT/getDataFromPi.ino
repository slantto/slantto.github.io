void getDataFromPi() {
  // receive data from PC and save it into inputBuffer

  if(Serial.available() > 0) {

    char x = Serial.read();

    // the order of these IF clauses is significant

    if (x == endMarker) {
      readInProgress = false;
      newDataFromPC = true;
      
      inputBuffer[bytesRecvd] = 0;
      parseData();
      Serial.println("<Command Parsed>");
    }

    if(readInProgress) {
      inputBuffer[bytesRecvd] = x;
      bytesRecvd ++;
      if (bytesRecvd == buffSize) {
        bytesRecvd = buffSize - 1;
      }
      Serial.println("<read in progress>");
    }

    if (x == startMarker) { 
      bytesRecvd = 0; 
      readInProgress = true;
      Serial.println("<Command recieved>");
    }
  }
}

void parseData(){
  //split commands from the pi into required components
  char* strtokIndx;
  strtokIndx = strtok(inputBuffer,","); //get first part 
  //messageFromPC = 
  strcpy(messageFromPC, strtokIndx); // copy it to messageFromPC


  strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
  switchCom = atoi(strtokIndx);     // convert this part to an integer

  strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
  COMPARAM1 = atof(strtokIndx);     // convert this part to an integer

  strtokIndx = strtok(NULL, ","); 
  COMPARAM2 = atof(strtokIndx);     // convert this part to a float

  strtokIndx = strtok(NULL, ","); 
  COMPARAM3 = atof(strtokIndx);     // convert this part to a float

}

