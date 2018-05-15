/*

*/
#include <Wire.h>
#include <WireData.h>
#include <OneWire.h>
#include <DallasTemperature.h>
//#include <PID_v1.h>

//define i2c slave address
#define slaveAddress 0x09


#define HeaterPin 33 //swtiches heater relay on or off
#define OilReturnValve 22 //
#define OilReturnPump 23
#define ExtractValve 24
#define ExtractPump  25
#define InjectValve 26
#define InjectPump 27
#define WaterOutValve 28
#define ExtractPolarityA 30
#define ExtractPolarityB 31

#define ONE_WIRE_BUS 7
float tempC;
double setTemp;
//double Input, Output;

//PID myPID(&Input, &Output, &setTemp,2,5,1, DIRECT);


// Setup a oneWire instance to communicate with any OneWire devices (not just Maxim/Dallas temperature ICs)
OneWire oneWire(ONE_WIRE_BUS);
 
// Pass our oneWire reference to Dallas Temperature. 
DallasTemperature sensors(&oneWire);
 
// arrays to hold device address
DeviceAddress tempSensor;

//I2c Address
int window = 5000;
unsigned long winstart;

void setup() {
  Serial.begin(9600);
  
    pinMode(OilReturnValve, OUTPUT);
    pinMode(OilReturnPump, OUTPUT);
    pinMode(ExtractValve, OUTPUT);
    pinMode(ExtractPump, OUTPUT);
    pinMode(InjectValve, OUTPUT);
    pinMode(InjectPump, OUTPUT);
    pinMode(WaterOutValve, OUTPUT);
    pinMode(ExtractPolarityA, OUTPUT);
    pinMode(ExtractPolarityB, OUTPUT);
    pinMode(HeaterPin, OUTPUT);
    digitalWrite(OilReturnValve, LOW);
    digitalWrite(OilReturnPump, LOW);
    digitalWrite(ExtractValve, LOW);
    digitalWrite(ExtractPump, LOW);
    digitalWrite(InjectValve, LOW);
    digitalWrite(InjectPump, LOW);
    digitalWrite(WaterOutValve, LOW);
    digitalWrite(ExtractPolarityA, HIGH);
    digitalWrite(ExtractPolarityB, HIGH);
    digitalWrite(HeaterPin, HIGH);//relay is normally closed
    
    sensors.begin();
    if (!sensors.getAddress(tempSensor, 0)){
      Serial.println("Sensor Error");
    }
 // sensors.setResolution(tempSensor, 9);
   //sensors.setWaitForConversion(false);
   
    sensors.setHighAlarmTemp(tempSensor, 110);
    sensors.setLowAlarmTemp(tempSensor, -10);
     Wire.begin(slaveAddress);  // I2C slave address 8 setup.
     Wire.onReceive(i2cReceive);  // register our handler function with the Wire library
     Wire.onRequest(i2cTransmit);  // register data return handler
     
     
  setTemp = 85;
  winstart = millis();

}

void loop() {
  sensors.requestTemperatures();
  tempC = sensors.getTempCByIndex(0);
  Serial.println(tempC);
 if(sensors.hasAlarm(tempSensor)){
    digitalWrite(HeaterPin,HIGH);
    Serial.println("TOO HOT!!! HEAT OFF!!!");
  }
else {
  if (tempC<setTemp-2){
    if (millis()-winstart>window){
      if (digitalRead(HeaterPin)==HIGH){
        digitalWrite(HeaterPin,LOW);
        Serial.print("on");
        winstart = millis();
      }
      else if (digitalRead(HeaterPin)==LOW){
        digitalWrite(HeaterPin,HIGH);
        winstart = millis();
      }
    }
    
  }
  else if (tempC>=setTemp-2){
        digitalWrite(HeaterPin, HIGH);
      }
  }
/*input = tempC;
  myPID.Compute();
  unsigned long now = millis();
  if(now - winstart>window){
    winstart += window }
  if(Output > now - winstart){ digitalWrite(HeaterPin,HIGH);};
  else {digitalWrite(HeaterPin,LOW);}
  */
  if (Serial.available()>0){
   int sercommand = Serial.read();
   switch(sercommand){
     case '1':
     if(digitalRead(InjectValve) == LOW){// || (digitalRead(InjectPump == LOW))){
       digitalWrite(InjectValve, HIGH);digitalWrite(InjectPump, HIGH);
   }
   else{
       digitalWrite(InjectValve, LOW);digitalWrite(InjectPump, LOW);
    }
    break;
    case '2':
    if(digitalRead(ExtractValve) == LOW) {//|| (digitalRead(ExtractPump == LOW))){
      digitalWrite(ExtractValve, HIGH);digitalWrite(ExtractPump, HIGH);
    }
   else{
       digitalWrite(ExtractValve, LOW);digitalWrite(ExtractPump, LOW);
    }
    break; 
    case '3':
    if(digitalRead(WaterOutValve) == LOW){
      digitalWrite(WaterOutValve, HIGH);digitalWrite(OilReturnPump, HIGH);
    }
    else{
       digitalWrite(WaterOutValve, LOW);digitalWrite(OilReturnPump, LOW);
    }
    break;
    case '4':
    if(digitalRead(OilReturnValve) == LOW) {//|| (digitalRead(OilReturnPump == LOW))){
      digitalWrite(OilReturnValve, HIGH);digitalWrite(OilReturnPump, HIGH);
    }
    else{
       digitalWrite(OilReturnValve, LOW);digitalWrite(OilReturnPump, LOW);
    }
    break;
    case '5':
    digitalWrite(OilReturnValve, HIGH);
    digitalWrite(OilReturnPump, HIGH);
    digitalWrite(ExtractValve, HIGH);
    digitalWrite(ExtractPump, HIGH);
    digitalWrite(InjectValve, HIGH);
    digitalWrite(InjectPump, HIGH);
    digitalWrite(WaterOutValve, HIGH);
    digitalWrite(ExtractPolarityA, HIGH);
    digitalWrite(ExtractPolarityB, HIGH);
    break;
    case '6':
    digitalWrite(OilReturnValve, LOW);
    digitalWrite(OilReturnPump, LOW);
    digitalWrite(ExtractValve, LOW);
    digitalWrite(ExtractPump, LOW);
    digitalWrite(InjectValve, LOW);
    digitalWrite(InjectPump, LOW);
    digitalWrite(WaterOutValve, LOW);
    digitalWrite(ExtractPolarityA, LOW);
    digitalWrite(ExtractPolarityB, LOW);
    //digitalWrite(HeaterPin, LOW);
    break;
    case '7':
    if (digitalRead(ExtractPolarityA) == HIGH){
      digitalWrite(ExtractPolarityA, LOW);
    digitalWrite(ExtractPolarityB, LOW);
   Serial.println("switch");
    }
    else {
      digitalWrite(ExtractPolarityA,HIGH);
    digitalWrite(ExtractPolarityB, HIGH);
    Serial.println("switch");
    }
    break;
    default:
    
    break;
   }
  }
}

void i2cReceive() {
  int command = Wire.read(); //should this be byte?
  
  switch(command){
    case 0x0A: //injection
    if(digitalRead(InjectValve) == LOW){// || (digitalRead(InjectPump == LOW))){
//      if (Wire.available() > 0){
//        int InjT = Wire.read(); //in seconds
//        InjT = InjT*1000;
//        digitalWrite(InjectValve, HIGH);digitalWrite(InjectPump, HIGH);
//        delay(InjT);
//        digitalWrite(InjectValve, LOW);digitalWrite(InjectPump, LOW);
//      }
      digitalWrite(InjectValve, HIGH);digitalWrite(InjectPump, HIGH);
    }
    else{
       digitalWrite(InjectValve, LOW);digitalWrite(InjectPump, LOW);
    }
    break;
    case 0x0B: //extraction
    if(digitalRead(ExtractValve) == LOW){ //|| (digitalRead(ExtractPump == LOW))){
//      if (Wire.available() > 0){
//        int InjT = Wire.read(); //in seconds
//        InjT = InjT*1000;
//        digitalWrite(ExtractValve, HIGH);digitalWrite(ExtractPump, HIGH);
//        delay(InjT);
//        digitalWrite(ExtractValve, LOW);digitalWrite(ExtractPump, LOW);
//      }
      digitalWrite(ExtractValve, HIGH);digitalWrite(ExtractPump, HIGH);
    }
    else{
       digitalWrite(ExtractValve, LOW);digitalWrite(ExtractPump, LOW);
    }
    break;    
    case 0x0C: //H2O out
    if(digitalRead(WaterOutValve) == LOW){
//      if (Wire.available() > 0){
//        int InjT = Wire.read(); //in seconds
//        InjT = InjT*1000;
//        digitalWrite(WaterOutValve, HIGH);digitalWrite(OilReturnPump, HIGH);
//        delay(InjT);
//        digitalWrite(WaterOutValve, LOW);digitalWrite(OilReturnPump, LOW);
//      }
      digitalWrite(WaterOutValve, HIGH);digitalWrite(OilReturnPump, HIGH);
    }
    else{
       digitalWrite(WaterOutValve, LOW);digitalWrite(OilReturnPump, LOW);
    }
    break;
    case 0x0D: //Oil return
    if(digitalRead(OilReturnValve) == LOW) {//|| (digitalRead(OilReturnPump == LOW))){
//      if (Wire.available() > 0){
//        int InjT = Wire.read(); //in seconds
//        InjT = InjT*1000;
//        digitalWrite(OilReturnValve, HIGH);digitalWrite(OilReturnPump, HIGH);
//        delay(InjT);
//        digitalWrite(OilReturnValve, LOW);digitalWrite(OilReturnPump, LOW);
//      }
      digitalWrite(OilReturnValve, HIGH);digitalWrite(OilReturnPump, HIGH);
    }
    else{
       digitalWrite(OilReturnValve, LOW);digitalWrite(OilReturnPump, LOW);
    }
    break;
    case 0x0E: //ALL ON
    digitalWrite(OilReturnValve, HIGH);
    digitalWrite(OilReturnPump, HIGH);
    digitalWrite(ExtractValve, HIGH);
    digitalWrite(ExtractPump, HIGH);
    digitalWrite(InjectValve, HIGH);
    digitalWrite(InjectPump, HIGH);
    digitalWrite(WaterOutValve, HIGH);
    digitalWrite(ExtractPolarityA, HIGH);
    digitalWrite(ExtractPolarityB, HIGH);
    //digitalWrite(Heater, HIGH);
    break;
    case 0x0F: //ALL OFF
    digitalWrite(OilReturnValve, LOW);
    digitalWrite(OilReturnPump, LOW);
    digitalWrite(ExtractValve, LOW);
    digitalWrite(ExtractPump, LOW);
    digitalWrite(InjectValve, LOW);
    digitalWrite(InjectPump, LOW);
    digitalWrite(WaterOutValve, LOW);
    digitalWrite(ExtractPolarityA, LOW);
    digitalWrite(ExtractPolarityB, LOW);
    //digitalWrite(Heater, LOW);
    break;
    
    default:
    /*digitalWrite(OilReturnValve, LOW);
    digitalWrite(OilReturnPump, LOW);
    digitalWrite(ExtractValve, LOW);
    digitalWrite(ExtractPump, LOW);
    digitalWrite(InjectValve, LOW);
    digitalWrite(InjectPump, LOW);
    digitalWrite(WaterOutValve, LOW);
    digitalWrite(ExtractPolarityA, LOW);
    digitalWrite(ExtractPolarityB, LOW);*/
    break;
  }
    
    
  
}

void i2cTransmit() {
  sensors.requestTemperatures();
  int tempOut = sensors.getTempCByIndex(0);
  Wire.write(tempOut);
  
}
