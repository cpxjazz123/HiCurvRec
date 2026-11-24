"""Serial full-data four-card diagnostics; never a formal training substitute."""
import os
from tools.research.stage2.launcher import launch
from tools.issue256.r02.identity import SOURCE, EVIDENCE, PYTHON


def main():
    for kind, entry in (('steady_profile', 'steady_profile_entry.py'),
                        ('validation', 'layer_validation_entry.py'),
                        ('benchmark', 'benchmark_entry.py')):
        code = launch(SOURCE / entry, EVIDENCE / kind / ('launcher_' + str(os.getpid())),
                      python=PYTHON, log_name=kind + '.log')
        print(kind, 'launcher_exit', code, flush=True)
        if code:
            return code
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
