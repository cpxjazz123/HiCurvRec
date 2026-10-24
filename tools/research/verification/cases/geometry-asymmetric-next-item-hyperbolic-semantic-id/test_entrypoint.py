import sys, unittest
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path("/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_hyperbolic_item_semantic_id")
sys.path.insert(0,str(ROOT.parent))
# Operator correction: the trainer does `from model import RQVAE`, a top-level
# import, so the tree directory itself has to be importable as well. The draft
# only put the parent on the path, so the suite could not even import the
# trainer it is meant to exercise.
sys.path.insert(0,str(ROOT))
from TIGER_c117_hyperbolic_item_semantic_id.model import RQVAE
from TIGER_c117_hyperbolic_item_semantic_id import train_rqvae as trainer

class TestTrainerEntrypointMechanism(unittest.TestCase):
    def test_source_entrypoint_and_real_candidate_optimizer_step(self):
        source=(ROOT/"train_rqvae.py").read_text()
        for required in ("model(batch)","raw_module.compute_loss(batch, reconstructed, quant_loss)","loss.backward()","optimizer.step()","DistributedSampler"):
            self.assertIn(required,source)
        self.assertEqual(trainer._tokenizer_config().geometry,"hyperbolic")
        cfg=SimpleNamespace(hidden_sizes=(8,),codebook_dim=4,dropout=0.,codebook_num=1,codebook_size=(4,),beta=.25,vq_type="vq",ema_decay=.99,fix_code_embs=False,sk_epsilon=.003,sk_iters=3,geometry="hyperbolic")
        model=RQVAE(cfg,in_dim=4)
        # Exercise the same forward -> compute_loss -> backward -> AdamW step used in main.
        x=torch.tensor([[.1,.3,.2,.7],[.8,.2,.5,.4]],dtype=torch.float32)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-2,weight_decay=1e-4)
        before=[p.detach().clone() for p in model.parameters()]
        opt.zero_grad(set_to_none=True)
        reconstructed,quant_loss,_,_=model(x)
        loss,_=model.compute_loss(x,reconstructed,quant_loss)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward(); opt.step()
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,model.parameters())),"trainer optimizer mechanism changed no parameters")
