"""Issue253 inclusive timing: target0.4, expected<=0.5, authorized ceiling0.8."""
import math
import statistics
from tools.issue252.performance import summarize_window as inclusive_window

MIN_EPOCHS = 30
TARGET_SECONDS = .4
EXPECTED_MAX_SECONDS = .5
MAX_MEAN_SECONDS = .8

def summarize_window(records):
    result = inclusive_window(records)
    # Assign every gap to the following epoch; all distributions use the same
    # enclosing wall scope as the mean, not optimizer-only epoch durations.
    durations = [float(r['end']) - (float(records[i-1]['end']) if i else float(r['start']))
                 for i, r in enumerate(records)]
    durations.sort()
    result['median_seconds'] = statistics.median(durations)
    result['p95_seconds'] = durations[math.ceil(.95*len(durations))-1]
    result['p95_definition'] = 'nearest-rank ceil(0.95*N)'
    result['distribution_scope'] = 'end-to-end epoch increments; interepoch work belongs to following epoch'
    result['target_seconds'] = TARGET_SECONDS
    result['expected_max_seconds'] = EXPECTED_MAX_SECONDS
    result['authorized_ceiling_seconds'] = MAX_MEAN_SECONDS
    result['target_met'] = result['epochs'] >= MIN_EPOCHS and result['mean_seconds'] <= EXPECTED_MAX_SECONDS
    result['passed'] = result['epochs'] >= MIN_EPOCHS and result['mean_seconds'] <= MAX_MEAN_SECONDS
    return result
