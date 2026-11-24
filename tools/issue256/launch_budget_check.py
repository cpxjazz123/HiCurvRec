"""Exercise full real-loss budget regression through the sole GPU launcher."""
from tools.research.stage2.launcher import launch
from tools.issue256.identity import SOURCE, EVIDENCE, PYTHON


def main():
    return launch(SOURCE / 'module_budget_entry.py', EVIDENCE / 'budget_regression_launcher',
                  python=PYTHON, log_name='budget_regression.log')


if __name__ == '__main__':
    raise SystemExit(main())
