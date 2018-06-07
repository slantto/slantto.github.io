void DRILL_OP_BUTT() {


  scale2.set_scale();
  scale2.tare();	//Reset the scale2 to 0

  while ((digitalRead(Y_MAX_PIN) == HIGH) && (digitalRead(Y_MIN_PIN) == HIGH)) {

    digitalWrite(Y_STEP_PIN, HIGH);
    delayMicroseconds(500);
    digitalWrite(Y_STEP_PIN, LOW);
    delayMicroseconds(500);

    if ((millis() - tstart) > 5000) {
      tstart = millis();
      readvalue = analogRead(CSENSE_PIN);
      Voltage = (readvalue * 5) / 1024;
      VRMS = (Voltage / 2.0) * 0.707;
      IRMS = (VRMS * 1000) / mVperAmp;
      Serial.print("<Arms= ");
      Serial.print(IRMS);
      Serial.println(" >");
      scale2.tare();
      scale2.set_scale(calibration_factor2); //Adjust to this calibration factor
      WOB2 = scale2.get_units();
      WOBavg = WOB2 * 4;
      Serial.print("<WOB = ");
      Serial.print(abs(WOBavg));
      Serial.println(" >");
    }

    if (digitalRead(DRILLD_BUTT) == LOW) {
      delay(200);
      if (digitalRead(DRILL_DIR_PIN) == LOW) {
        digitalWrite(DRILL_DIR_PIN, HIGH);
        Serial.println("<DRILL FORWARD!>");
      }
      else if (digitalRead(DRILL_DIR_PIN) == HIGH) {
        digitalWrite(DRILL_DIR_PIN, LOW);
        Serial.println("<DRILL REVERSE!>");
      }
    }

    if (digitalRead(SCREWD_BUTT) == LOW) {
      delay(200);
      if (digitalRead(Y_DIR_PIN) == LOW) {
        digitalWrite(Y_DIR_PIN, HIGH);
        Serial.println("<DRILL DOWN!>");
      }
      else if (digitalRead(Y_DIR_PIN) == HIGH) {
        digitalWrite(Y_DIR_PIN, LOW);
        Serial.println("<DRILL UP!>");
      }
    }

    if (digitalRead(EXTRA_BUTT) == LOW) {
      delay(200);
      Serial.println("<PAUSE>");
      while (digitalRead(EXTRA_BUTT) == HIGH) {
        if ((millis() - tstart) > 5000) {
          tstart = millis();
          readvalue = analogRead(CSENSE_PIN);
          Voltage = (readvalue * 5) / 1024;
          VRMS = (Voltage / 2.0) * 0.707;
          IRMS = (VRMS * 1000) / mVperAmp;
          Serial.print("<Arms= ");
          Serial.print(IRMS);
          Serial.println(" >");
          scale2.tare();
          scale2.set_scale(calibration_factor2); //Adjust to this calibration factor
          WOB2 = scale2.get_units();
          WOBavg = WOB2 * 4;
          Serial.print("<WOB = ");
          Serial.print(abs(WOBavg));
          Serial.println(" >");
        }
      }
    }
    if (digitalRead(DRILLOP_BUTT) == LOW) {
      delay(200);
      Serial.println("<EXITING>");
      break;
    }
  }
}

