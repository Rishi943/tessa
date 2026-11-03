-- A visitor whose star cleared the bar may leave an email, so we can write when Tessa's own checks have run.
-- Write-only from the site: no route or Ask Tessa tool reads this table.
CREATE TABLE follow (
  tic INTEGER NOT NULL,
  email TEXT NOT NULL,
  at TEXT NOT NULL,
  PRIMARY KEY (tic, email)
);
