"""Reproduce frozen-batch GNN tape growth using the actual reconstruction loss."""
from pathlib import Path
import json
import subprocess
import sys
import unittest

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())
PYTHON='/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9'


def probe():
    import numpy as np
    import torch
    from unittest.mock import patch
    tree=ROOT/'stage2_RQ-VAE/_archive/TIGER_gnn_attention_sectional_rq'
    sys.path.insert(0,str(tree))
    import train_rqvae as training
    import assignment_floor
    torch.set_num_threads(1);torch.manual_seed(42)
    model=training.RQVAE(training._tokenizer_config(),in_dim=768,distance=training.DISTANCE)
    original=torch.load(ROOT/'results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_best.pth',map_location='cpu',weights_only=False)
    model.load_state_dict(original['state_dict'],strict=False)
    catalogue=torch.from_numpy(np.load(training.EMBEDDING_FILE))
    model.freeze_tangent_scale(catalogue)
    ids=torch.arange(32);batch=catalogue[ids]
    history={key:value.clone() for key,value in assignment_floor._COUNTS.items()}
    def run():
        assignment_floor._COUNTS.clear()
        assignment_floor._COUNTS.update({key:value.clone() for key,value in history.items()})
        model.zero_grad(set_to_none=True);saved=0
        def pack(tensor):
            nonlocal saved
            if tensor.is_floating_point():saved+=tensor.numel()
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda tensor:tensor):
            reconstruction,parts,_,tokens=model(batch,ids)
            loss,_=training.CRITERION(reconstruction,batch,parts)
        loss.backward()
        gradients={name:p.grad.clone() for name,p in model.named_parameters() if p.grad is not None}
        return saved,float(loss.detach()),tokens.detach().clone(),gradients
    bounded=run()
    # This assertion fails on the original real-loss path, before any mock.
    print(json.dumps({'actual_loss_saved_float_elements':bounded[0],
                      'actual_loss':bounded[1],'maximum_saved_float_elements':5000000}),flush=True)
    assert bounded[0]<5000000,'expanded ego-node GNN activations remain on the frozen-batch training tape'
    with patch('model.structure.checkpoint',side_effect=lambda function,*args,**kwargs:function(*args)):
        direct=run()
    assert direct[0]>bounded[0]*2,'the memory regression must distinguish direct activation retention'
    assert bounded[1]==direct[1] and torch.equal(bounded[2],direct[2]),'checkpointing changed hard forward/loss'
    assert bounded[3].keys()==direct[3].keys(),'checkpointing changed gradient routing'
    for name,gradient in bounded[3].items():
        torch.testing.assert_close(gradient,direct[3][name],rtol=1e-5,atol=1e-7,msg=name)
    assignment_floor._COUNTS.clear();assignment_floor._COUNTS.update(history)
    print(json.dumps({'status':'PASS','checkpoint_saved_floats':bounded[0],
                      'direct_saved_floats':direct[0],'actual_loss':bounded[1],
                      'gradient_parameters':len(bounded[3])}),flush=True)


class ActivationMemoryTests(unittest.TestCase):
    def test_real_loss_tape_bound_and_gradient_equivalence(self):
        result=subprocess.run([PYTHON,str(Path(__file__).resolve()),'--probe'],cwd=ROOT,
                              capture_output=True,text=True,timeout=900)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        print(result.stdout,end='')


if __name__=='__main__':
    if sys.argv[1:]==['--probe']:probe()
    else:unittest.main()
