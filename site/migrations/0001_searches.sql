-- Visitors' searches. results holds the first finished search of each star; claims keeps two visitors searching
-- at once off the same star; totals is one row, so the stars-left count never scans the table.
CREATE TABLE results (
  tic INTEGER PRIMARY KEY,
  pass INTEGER NOT NULL,
  handle TEXT NOT NULL DEFAULT '',
  at TEXT NOT NULL,
  row TEXT NOT NULL
);
CREATE INDEX results_pass ON results (pass) WHERE pass = 1;
CREATE TABLE claims (
  tic INTEGER PRIMARY KEY,
  until INTEGER NOT NULL
);
CREATE TABLE totals (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  searched INTEGER NOT NULL,
  passed INTEGER NOT NULL
);
INSERT INTO totals VALUES (1, 0, 0);
