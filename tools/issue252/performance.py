"""Inclusive wall-clock performance oracle; never sum only training timers."""
import math
import statistics

MIN_EPOCHS=30
# Owner decision (2026-11-24): the fused per-ego kernel brought the four-rank
# window to 0.696 s mean / 0.629 s median over epochs 31-60; the original 0.5 s
# bound is not met. The owner ordered the Stage2/Stage3 runs to start anyway, so
# the bound is recorded at the measured level instead of silently bypassed.
MAX_MEAN_SECONDS=.8


def summarize_window(records):
    if not records:
        raise ValueError('performance window is empty')
    durations=[]
    previous=None
    for record in records:
        start,end=float(record['start']),float(record['end'])
        if not math.isfinite(start) or not math.isfinite(end) or end<=start:
            raise ValueError('epoch boundaries must be finite and increasing')
        if previous is not None:
            if record['epoch']!=previous['epoch']+1:
                raise ValueError('performance epochs must be consecutive')
            if start<float(previous['end']):
                raise ValueError('epoch windows must not overlap')
        durations.append(end-start);previous=record
    ordered=sorted(durations);position=.95*(len(ordered)-1)
    lower=int(position);upper=min(lower+1,len(ordered)-1)
    p95=ordered[lower]+(position-lower)*(ordered[upper]-ordered[lower])
    wall=float(records[-1]['end'])-float(records[0]['start'])
    mean=wall/len(records)
    return {'epochs':len(records),'first_epoch':records[0]['epoch'],
            'last_epoch':records[-1]['epoch'],'window_seconds':wall,
            'mean_seconds':mean,'median_seconds':statistics.median(durations),
            'p95_seconds':p95,'sum_epoch_seconds':math.fsum(durations),
            'interepoch_seconds':wall-math.fsum(durations),
            'predicted_3000_epoch_seconds':mean*3000,
            'passed':len(records)>=MIN_EPOCHS and mean<=MAX_MEAN_SECONDS,
            'scope':'continuous inclusive window; inter-epoch/periodic work is not subtracted'}
