void BUTT_STATE() {
  //Reads buttons, does things

  if (digitalRead(HOT_BUTT) == LOW) {
    if (digitalRead(HEATER_0_PIN) == LOW) {
      digitalWrite(HEATER_0_PIN, HIGH);
      Serial.println("<Heater ON!>");
    }
    else if (digitalRead(HEATER_0_PIN) == HIGH) {
      digitalWrite(HEATER_0_PIN, LOW);
      Serial.println("<Heater OFF!>");
    }
    return;
  }

  else if (digitalRead(DRILLD_BUTT) == LOW) {
    if (digitalRead(DRILL_DIR_PIN) == LOW) {
      digitalWrite(DRILL_DIR_PIN, HIGH);
      Serial.println("<DRILL FORWARD!>");
    }
    else if (digitalRead(DRILL_DIR_PIN) == HIGH) {
      digitalWrite(DRILL_DIR_PIN, LOW);
      Serial.println("<DRILL REVERSE!>");
    }
    return;
  }

  else if (digitalRead(SCREWD_BUTT) == LOW) {
    if (digitalRead(Y_DIR_PIN) == LOW) {
      digitalWrite(Y_DIR_PIN, HIGH);
      Serial.println("<DRILL DOWN!>");
    }
    else if (digitalRead(Y_DIR_PIN) == HIGH) {
      digitalWrite(Y_DIR_PIN, LOW);
      Serial.println("<DRILL UP!>");
    }
    return;
  }

  else if (digitalRead(DRILLOP_BUTT) == LOW) {
    DRILL_OP_BUTT();
    return;
  }

  else if (digitalRead(ZOP_BUTT) == LOW) {
    DRILLZ_BUTT();
    return;
  }

  else if (digitalRead(ZDIR_BUTT) == LOW) {
    if (digitalRead(Z_DIR_PIN) == LOW) {
      digitalWrite(Z_DIR_PIN, HIGH);
    }
    else if (digitalRead(Z_DIR_PIN) == HIGH) {
      digitalWrite(Z_DIR_PIN, LOW);
    }
    return;
  }

  else if (digitalRead(XOP_BUTT) == LOW) {
    DRILLX_BUTT();
    return;
  }

  else if (digitalRead(XDIR_BUTT) == LOW) {
    if (digitalRead(X_DIR_PIN) == LOW) {
      digitalWrite(X_DIR_PIN, HIGH);
    }
    else if (digitalRead(X_DIR_PIN) == HIGH) {
      digitalWrite(X_DIR_PIN, LOW);
    }
    return;
  }

  else if (digitalRead(EXTRA_BUTT) == LOW) {
    
  }
}

