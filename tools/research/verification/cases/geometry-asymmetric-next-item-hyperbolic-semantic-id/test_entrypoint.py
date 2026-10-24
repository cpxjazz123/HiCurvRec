import sys, unittest
from pathlib import Path
import torch
ROOT=Path("/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_hyperbolic_item_semantic_id")
sys.path.insert(0,str(ROOT))
import train_rqvae as trainer
from model import RQVAE

class TestTrainerEntrypointMechanism(unittest.TestCase):
    def test_candidate_trainer_config_forward_loss_backward_and_optimizer_step(self):
        torch.manual_seed(9)
        cfg=trainer._tokenizer_config()
        self.assertEqual(cfg.geometry,"hyperbolic")
        self.assertEqual(cfg.codebook_num,3)
        model=RQVAE(cfg,in_dim=6)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=trainer.WEIGHT_DECAY)
        batch=torch.tensor([[.1,.2,.3,.4,.5,.6],[.6,.5,.4,.3,.2,.1],[.2,.4,.1,.6,.3,.5]],dtype=torch.float32)
        before=[p.detach().clone() for p in model.parameters()]
        opt.zero_grad(set_to_none=True)
        reconstructed,quant_loss,_,_=model(batch)
        loss,recon_loss=model.compute_loss(batch,reconstructed,quant_loss)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.parameters()))
        torch.nn.utils.clip_grad_norm_(model.parameters(),trainer.GRADIENT_CLIP_NORM)
        opt.step()
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,model.parameters())),"trainer optimizer step had no effect")
        source=(ROOT/"train_rqvae.py").read_text()
        self.assertIn("reconstructed, quant_loss, _, _ = model(batch)",source)
        self.assertIn("raw_module.compute_loss(batch, reconstructed, quant_loss)",source)
        self.assertIn("loss.backward()",source)
        self.assertIn("optimizer.step()",source)
