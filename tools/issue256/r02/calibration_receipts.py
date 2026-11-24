"""Shared audit of complete native-block calibration records."""
import math


def validate(records, *, epochs, interval, steps_per_epoch, checkpoint_weight=None):
    if [row.get('calibration_epoch') for row in records] != list(range(1,epochs+1,interval)):
        raise ValueError('calibration ledger must contain all complete native anchors')
    for row in records:
        if row.get('calibration_step') != (row['calibration_epoch']-1)*steps_per_epoch:
            raise ValueError('anchor did not use its actual first step')
        weight=row['lambda']
        if not math.isfinite(weight) or weight<=0:
            raise ValueError('anchor requires a finite positive coefficient')
        budgets=row.get('module_budgets',{})
        if row.get('calibration_world_size')!=4 or row.get('calibration_batch_items')!=1024 or len(budgets)!=17:
            raise ValueError('anchor is not a complete four-rank17-module real batch')
        ratios=[]
        for budget in budgets.values():
            a,b=budget['original_gradient'],budget['curvature_gradient']
            if not all(math.isfinite(x) and x>=0 for x in (a,b)):
                raise ValueError('module norms must be finite nonnegative')
            if a==0 and b>0:
                raise ValueError('degenerate active original gradient')
            ratio=weight*b/a if a>0 else 0.
            if not math.isclose(budget['weighted_ratio'],ratio,rel_tol=2e-6,abs_tol=1e-12):
                raise ValueError('recorded module ratio differs from real norms and coefficient')
            if b>0:
                ratios.append(ratio)
        if not ratios or abs(max(ratios)-1)>2e-6:
            raise ValueError('anchor does not solve the maximal module budget')
    if checkpoint_weight is not None and float(checkpoint_weight)!=records[-1]['lambda']:
        raise ValueError('checkpoint differs from the last anchor coefficient')


def validate_diagnostic(report):
    validate(report['actual_calibration_records'],epochs=60,interval=50,steps_per_epoch=6)
    anchors=report.get('anchors',[])
    if [r.get('epoch') for r in anchors]!=[1,51] or [r.get('step') for r in anchors]!=[0,300]:
        raise ValueError('diagnostic must measure both actual native anchors')
    for anchor in anchors:
        budget=anchor.get('independent_global_budget',{})
        weights=budget.get('rank_lambdas',[])
        if not budget.get('passed') or len(budget.get('modules',{}))!=17 or weights!=[anchor['lambda']]*4:
            raise ValueError('independent four-rank current-state budget is incomplete')
        if not all(anchor.get(k) for k in ('current_sampler_population_exact',
            'current_structural_sample_exact','parameters_unchanged','coverage_unchanged')):
            raise ValueError('current-batch or non-updating calibration evidence failed')
    if not anchors[1].get('stale_coefficient_mutation',{}).get('caught'):
        raise ValueError('actual stale-coefficient mutation was not caught')
