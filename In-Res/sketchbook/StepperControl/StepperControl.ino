// defines pins numbers
const int stepPin = 3; 
const int dirPin = 4; 
 
void setup() {
  // Sets the two pins as Outputs
  pinMode(stepPin,OUTPUT); 
  pinMode(dirPin,OUTPUT);
  Serial.begin(9600);
  Serial.println("Enter number of revolutions(whole numbers only)(+ for CW,- for CCW)");
}
void loop() {
  if (Serial.available())
  {
    double rev =Serial.read();
    if (rev < 0){
      digitalWrite(dirPin,LOW);
    }
    else if (rev >= 0){
        digitalWrite(dirPin,HIGH);
    }
    
  
  // 20000 steps per revolution
  for(int x = 0; x < rev*20000; x++) {
    digitalWrite(stepPin,HIGH); 
    delayMicroseconds(500); 
    digitalWrite(stepPin,LOW); 
    delayMicroseconds(500); 
  }
  Serial.println("DONE!");
  delay(1000); // One second delay
  }
}
  
