## C9: validation launch refused before starting ranks; interpreter corrected

Command `python3 -B -m tools.issue255.launch_validation` exited 1 before creating a GPU run:

```text
LaunchRefused: Stage2 needs 4 visible cards for four-rank DDP and this host shows 0
```

Diagnosis on `m3g118`: system `python3` is `/apps/anaconda/2024.02-1/bin/python3`, with `ModuleNotFoundError: No module named 'torch'`. The launcher's visibility probe uses its controller interpreter. `nvidia-smi -L` lists four L40S; the required Stage2 interpreter reports PyTorch `2.8.0+cu128`, CUDA12.8 and four devices.

Corrected command uses `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -B -m tools.issue255.launch_validation`. No visibility override, launcher change, protocol change or CPU substitute. Its actual four-rank result is pending; C9 is not yet PASS. No corrected formal run has started.
