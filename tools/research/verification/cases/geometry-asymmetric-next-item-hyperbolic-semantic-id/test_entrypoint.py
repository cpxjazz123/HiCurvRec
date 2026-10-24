import sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np, pandas as pd, torch
TREE=Path("/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_hyperbolic_item_semantic_id")
sys.path.insert(0,str(TREE))
import train_rqvae as trainer
from model.layers import HyperbolicRQLayer

class RealTrainerEntrypoint(unittest.TestCase):
    def test_main_trains_saves_checkpoint_and_exports_real_sid(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td); emb=np.array([[.1,.2,.3],[.3,.1,.2],[.7,.2,.1],[.2,.8,.3]],dtype=np.float32)
            ep=d/'emb.npy'; np.save(ep,emb)
            frame=pd.DataFrame({'target':np.arange(len(emb),dtype=np.int64)})
            ck=d/'checkpoint.pth'; sidpath=d/'sid.npy'; js=d/'sid.json'; metrics=d/'metrics.jsonl'
            seen=[]; original=trainer.RQVAE.compute_loss
            def watched(model,*args,**kwargs):
                seen.append((model.geometry,isinstance(model.rq,HyperbolicRQLayer)))
                return original(model,*args,**kwargs)
            with patch.object(trainer,'EMBEDDING_FILE',ep), patch.object(trainer,'TRAIN_FILE',d/'train.parquet'), \
                 patch.object(trainer.pd,'read_parquet',return_value=frame), patch.object(trainer,'CHECKPOINT',ck), \
                 patch.object(trainer,'OUTPUT_SID',sidpath), patch.object(trainer,'OUTPUT_JSON',js), \
                 patch.object(trainer,'METRICS_PATH',metrics), patch.object(trainer,'EPOCHS',1), \
                 patch.object(trainer,'BATCH_SIZE_PER_RANK',4), patch.object(trainer,'HIDDEN_SIZES',(8,)), \
                 patch.object(trainer,'CODEBOOK_NUM',1), patch.object(trainer,'CODEBOOK_SIZE',(2,)), \
                 patch.object(trainer,'CODEBOOK_DIM',3), patch.object(trainer,'WARMUP_EPOCHS',0), \
                 patch.object(trainer,'EVAL_INTERVAL',1), patch.object(trainer,'SK_EPSILON',-1.), \
                 patch.object(trainer,'SK_ITERS',-1), patch.object(trainer.RQVAE,'compute_loss',watched):
                trainer.main()
            self.assertTrue(seen and all(g=='hyperbolic' and h for g,h in seen), 'real loss path did not use hyperbolic RQ')
            self.assertTrue(ck.is_file()); self.assertTrue(sidpath.is_file()); self.assertTrue(js.is_file())
            saved=torch.load(ck,map_location='cpu',weights_only=False)
            self.assertEqual(saved['epoch'],1); self.assertEqual(saved['selection'],'final_epoch')
            self.assertTrue(any('rq.vq_layers.0' in k for k in saved['state_dict']))
            sid=np.load(sidpath)
            self.assertEqual(sid.shape,(len(emb),2)); self.assertEqual(len(np.unique(sid,axis=0)),len(emb))
            exported=__import__('json').loads(js.read_text())
            # Operator correction: assertEqual on numpy arrays raises "truth value of an
            # array is ambiguous"; the same matrix equality is asserted explicitly here.
            np.testing.assert_array_equal(np.asarray([exported[str(i)] for i in range(len(emb))]),np.asarray(sid))
            # Independently reload the exact checkpoint and verify its raw codes are the SID prefix.
            cfg=trainer._tokenizer_config(); cfg.hidden_sizes=(8,); cfg.codebook_num=1; cfg.codebook_size=(2,); cfg.codebook_dim=3; cfg.sk_epsilon=-1.; cfg.sk_iters=-1
            model=trainer.RQVAE(cfg,in_dim=emb.shape[1]); model.load_state_dict(saved['state_dict']); model.eval()
            raw=model.get_indices(torch.from_numpy(emb),infer_use_sk=False).numpy()
            np.testing.assert_array_equal(sid[:,:-1],raw)
