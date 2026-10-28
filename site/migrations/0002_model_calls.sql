-- One row per model call from the site (Ask Tessa), so the demo can show which model does how much of
-- Tessa's work; serve/usage.py adds these to the pipeline's own log.
CREATE TABLE model_calls (
  at TEXT NOT NULL,
  model TEXT NOT NULL,
  purpose TEXT NOT NULL,
  input INTEGER NOT NULL,
  output INTEGER NOT NULL
);
