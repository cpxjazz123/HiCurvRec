"""A valid greedy floor is not an optimal-value Lipschitz oracle."""
from pathlib import Path
import torch
import pytest
import validation_entry
from model.layers import VQLayer


def make_case():
    with torch.random.fork_rng(devices=[]):
        layer=VQLayer(2,32)
    layer.eval()
    left=torch.tensor([[0.,2.],[1.,2.0009765625]])
    right=torch.tensor([[0.,2.],[1.,1.9990234375]])
    left_ids=layer._indices(left,False).reshape(2,1)
    right_ids=layer._indices(right,False).reshape(2,1)
    assert left_ids.tolist()==[[1],[0]], 'frozen greedy repair left hand answer'
    assert right_ids.tolist()==[[0],[1]], 'frozen greedy repair right hand answer'
    observer=validation_entry.Observer({},Path('.'))
    observer.assignment_layers=[layer]
    return observer,left,right,left_ids,right_ids


def test_valid_frozen_greedy_allocations_are_not_rejected_as_nonoptimal():
    observer,left,right,left_ids,right_ids=make_case()
    result=observer.auxiliary_ties(left_ids,right_ids,[left],[right],balanced=False)
    assert result['levels'][0]['shared_objective']==3.0
    assert result['levels'][0]['independent_objective']==1.9990234375
    assert result['differing_items']==2, 'both valid greedy IDs change despite a tiny cost perturbation'


def test_wrong_recorded_hard_codes_cannot_pass_replay():
    observer,left,right,left_ids,right_ids=make_case()
    with pytest.raises(AssertionError,match='recorded hard IDs do not replay'):
        observer.auxiliary_ties(torch.zeros_like(left_ids),right_ids,[left],[right],balanced=False)


@pytest.mark.parametrize('balanced,expected', [(False,[[1],[0]]),(True,[[0],[1]])])
def test_replay_uses_captured_population_mode_not_restored_model_training(balanced,expected):
    layer=VQLayer(2,32,sk_epsilon=.003,sk_iters=50)
    cost=torch.tensor([[0.,2.],[1.,2.0009765625]])
    layer.train(balanced)
    recorded=layer._indices(cost,False).reshape(2,1)
    assert recorded.tolist()==expected, 'captured native allocation hand answer'
    layer.train()  # quantize restores the enclosing model state after either population.
    observer=validation_entry.Observer({},Path('.'))
    observer.assignment_layers=[layer]
    result=observer.auxiliary_ties(recorded,recorded,[cost],[cost],balanced=balanced)
    assert result['levels'][0]['exact_native_rule_replay'], 'each population must replay its captured native mode'
    assert layer.training and layer.use_sk, 'replay must restore the live layer allocation state'
