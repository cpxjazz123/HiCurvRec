# Diagnostic launch failure before GPU work

Observed command: `python3 -m tools.issue252.launch_optimized_profile`, exit1.

```
File "tools/issue252/launch_optimized_profile.py", line 14, in main
    return launch(ENTRY, OUTPUT / ('launcher_' + str(os.getpid())), python=PYTHON,
File "tools/research/stage2/launcher.py", line 108, in launch
    assert_four_cards(_visible_cards(env_extra))
tools.research.stage2.launcher.LaunchRefused: Stage2 needs 4 visible cards for four-rank DDP and this host shows 0; a single-card run is not an accepted Stage2 regime
OPTIMIZED_PROFILE_LAUNCH_EXIT=1
```

This controller was invoked with system Python, unlike the successful original profile's pinned Stage2 interpreter. The launcher probes torch in the controller before spawning its explicit child interpreter. Inspect the controller dependency in CONTROLLER_INTERPRETER.json; do not interpret this refusal as measured absence of hardware, bypass the four-card guard, or count it as a training attempt. The sanctioned correction is to use `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m tools.issue252.launch_optimized_profile` as required by repository interpreter policy. New runtime results remain pending.
