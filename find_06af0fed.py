"""Find which hrqvae ckpt produced SID 06af0fed by hashing sid_output.npy."""
import os, sys, glob, hashlib
import numpy as np

target_sha = "06af0fedf4907b05b2e1efd45ab6a96dd4aa36fa14db05b2c8f285283b7f1944"
root = "/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history"
for sid_path in glob.glob(os.path.join(root, "*", "sid_output.npy")):
    h = hashlib.sha256()
    with open(sid_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    sha = h.hexdigest()
    if sha == target_sha:
        print(f"MATCH: {sid_path}")
        sid = np.load(sid_path)
        print(f"  shape={sid.shape}, dtype={sid.dtype}")
        # 找最近 stage2 ckpt (同目录)
        sid_dir = os.path.dirname(sid_path)
        ckpt_path = os.path.join(sid_dir, "hrqvae_kappa_sync.ckpt")
        print(f"  ckpt exists: {os.path.exists(ckpt_path)}")
print("\n[done]")