"""The one authorized four-card path for the60epoch refresh regression."""
import os
from tools.research.stage2.launcher import launch
from tools.issue256.r04.identity import SOURCE,EVIDENCE,PYTHON


def main():
    return launch(SOURCE/'periodic_budget_entry.py',
        EVIDENCE/'periodic_budget_launcher'/str(os.getpid()),python=PYTHON,log_name='periodic.log')


if __name__=='__main__':
    raise SystemExit(main())
