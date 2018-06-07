void DRILLZ_BUTT() {
  //runs zmotors 

  while ((digitalRead(Z_MAX_PIN) == HIGH) && (digitalRead(Z_MIN_PIN) == HIGH)) {

    digitalWrite(Z_STEP_PIN, HIGH);
    delayMicroseconds(500);
    digitalWrite(Z_STEP_PIN, LOW);
    delayMicroseconds(500);

    if (digitalRead(ZDIR_BUTT) == LOW) {
      if (digitalRead(Z_DIR_PIN) == LOW) {
        digitalWrite(Z_DIR_PIN, HIGH);
      }
      else if (digitalRead(Z_DIR_PIN) == HIGH) {
        digitalWrite(Z_DIR_PIN, LOW);
      }
    }

    if (digitalRead(EXTRA_BUTT) == LOW) {
      delay(2000);
      while (digitalRead(EXTRA_BUTT) == HIGH) {

      }
    }

    if (digitalRead(ZOP_BUTT) == LOW) {
      break;
    }

  }

}

