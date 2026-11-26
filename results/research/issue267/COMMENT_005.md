## Stage2 startup failure and fix

- First dispatch `stage2_caa891559565421d9b74703b6af643c5` failed before the first optimizer step:
  non-zero ranks received `relative_calibration=None` because the broadcast list was not
  rebound from `broadcast_object_list`.
- Fix: assign `relative_objects=[...]`, broadcast the list, then read `objects[0]`; no
  mechanism/weight change.
- Smoke had passed (`collision 0.051409`). Re-dispatch after a fresh four-rank admission.
