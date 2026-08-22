import argparse
import os
import gin
import torch
from data.schemas import TokenizedSeqBatch


def eval_mode(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out

    return inner


def parse_config():
    # 优先 HGREC_TAG env var 推导 config_path (相对路径, 不需要 CLI 传入)
    hgrec_tag = os.environ.get("HGREC_TAG")
    if hgrec_tag:
        config_path = f"configs/decoder_instruments_hgrec_{hgrec_tag}.gin"
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("config_path", type=str, help="Path to gin config file.")
        args = parser.parse_args()
        config_path = args.config_path
    try:
        gin.parse_config_file(config_path)
    except Exception as e:
        # HG-Rec 路径兜底: gin binding 失败时靠 FORCE_HGREC 环境变量 + sys.argv 检测
        print(f"[parse_config] gin binding failed ({e}); falling back to env/argv", flush=True)


@torch.no_grad
def compute_debug_metrics(
    batch: TokenizedSeqBatch, model_output=None, prefix: str = ""
) -> dict:
    seq_lengths = batch.seq_mask.sum(axis=1).to(torch.float32)
    prefix = prefix + "_"
    debug_metrics = {
        prefix + f"seq_length_p{q}": torch.quantile(seq_lengths, q=q)
        .detach()
        .cpu()
        .item()
        for q in [0.25, 0.5, 0.75, 0.9, 1]
    }
    if model_output is not None:
        loss_debug_metrics = {
            prefix + f"loss_{d}": model_output.loss_d[d].detach().cpu().item()
            for d in range(batch.sem_ids_fut.shape[1])
        }
        debug_metrics.update(loss_debug_metrics)
    return debug_metrics
