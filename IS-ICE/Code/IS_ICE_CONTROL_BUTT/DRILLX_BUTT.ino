void DRILLX_BUTT() {
  //runs x motors switches are reversed for some stupid reason

  while ((digitalRead(X_MAX_PIN) == LOW) && (digitalRead(X_MIN_PIN) == LOW)) {

    digitalWrite(X_STEP_PIN, HIGH);
    delayMicroseconds(500);
    digitalWrite(X_STEP_PIN, LOW);
    delayMicroseconds(500);

    if (digitalRead(XDIR_BUTT) == LOW) {
      if (digitalRead(X_DIR_PIN) == LOW) {
        digitalWrite(X_DIR_PIN, HIGH);
      }
      else if (digitalRead(X_DIR_PIN) == HIGH) {
        digitalWrite(X_DIR_PIN, LOW);
      }
    }

    if (digitalRead(EXTRA_BUTT) == LOW) {
      delay(2000);
      while (digitalRead(EXTRA_BUTT) == HIGH) {

      }
    }

    if (digitalRead(XOP_BUTT) == LOW) {
      break;
    }

  }

}

