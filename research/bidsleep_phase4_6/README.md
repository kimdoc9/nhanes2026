# BID-Sleep Phase 5–6 public execution

This isolated branch executes the pre-model stages for the study **Defining the Operating Envelope of Multi-Night Smartwatch Sleep-Continuity Monitoring Under Sensor Degradation**.

The workflow:

- retrieves the official PhysioNet BID-Sleep manifest;
- verifies 47 participants, 253 participant-nights and 759 expected data files;
- downloads one subject-night at a time;
- verifies every file against the official SHA-256 manifest;
- processes headerless Apple Watch IHR and accelerometry CSV files into compact 30-second features;
- deletes raw data after each night;
- applies automated structural QC without manual signal review or adjudication;
- derives expert- and Dreem-referenced TST, WASO, wake-bout burden and sleep–wake transition metrics;
- quantifies expert–Dreem agreement and repeated-night reference reliability;
- evaluates the locked endpoint-distribution gate before any model training.

No raw BID-Sleep data are committed to GitHub. Raw files exist only transiently on GitHub-hosted runners. Workflow artifacts contain compact derived features, QC records and reference metrics.

This branch does not perform model training, insomnia diagnosis, clinical validation or modification of the repository's main branch.
