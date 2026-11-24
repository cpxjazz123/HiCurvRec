"""Normalize the original diagnostic, retaining raw reports and their producer."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())
DIRECTORY=ROOT/'results/research/issue252/profiling_baseline'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    reports=[json.loads((DIRECTORY/f'rank_{rank}.json').read_text()) for rank in range(4)]
    for rank,report in enumerate(reports):
        if (report['status']!='PROFILED' or report['rank']!=rank or report['world_size']!=4
                or report['global_batch']!=4096 or report['completed_optimizer_steps']!=1
                or report['completed_epochs']!=0 or report['formal_training']):
            raise ValueError('not the frozen full-sized four-rank diagnostic')
    source_maps=[report['source_hashes'] for report in reports]
    if any(value!=source_maps[0] for value in source_maps):
        raise ValueError('ranks executed different source generations')
    snapshot=DIRECTORY/'source_snapshot.tar.gz'
    with tarfile.open(snapshot,'r:gz') as archive:
        for name,expected in source_maps[0].items():
            member=archive.extractfile(name)
            if member is None or hashlib.sha256(member.read()).hexdigest()!=expected:
                raise ValueError('executed snapshot mismatch: '+name)
    groups=defaultdict(list)
    ranks=[]
    for report in reports:
        for phase in report['phases']:
            groups[(phase['stage'],phase['phase'])].append({'rank':report['rank'],**phase})
        operators=[]
        for original in report['cuda_operators']:
            row=dict(original)
            # Legacy producer v1 divided CPU us but accidentally labelled native
            # device us as seconds. Raw records/source remain immutable evidence.
            if report.get('operator_duration_unit') is None:
                row['self_device_seconds']/=1e6
            elif report['operator_duration_unit']!='seconds':
                raise ValueError('unknown operator duration unit')
            operators.append(row)
        ranks.append({key:report[key] for key in ('rank','run_id','worker_pid','master_port','gpu','torch','cuda_build',
                      'initialization_seconds','initialization_peak_bytes','actual_one_step_wall_seconds',
                      'training_step_peak_bytes','loader')})
        ranks[-1]['communication']=[op for op in operators if any(word in op['name'].lower() for word in ('nccl','allreduce','c10d'))]
    phases=[]
    for (stage,name),rows in sorted(groups.items()):
        phases.append({'stage':stage,'phase':name,'ranks_present':[row['rank'] for row in rows],
            'calls_per_rank':{str(row['rank']):row['calls'] for row in rows},
            'cpu_dispatch_wall_s_min':min(row['cpu_dispatch_wall_seconds'] for row in rows),
            'cpu_dispatch_wall_s_max':max(row['cpu_dispatch_wall_seconds'] for row in rows),
            'cuda_stream_span_s_min':min(row['cuda_stream_span_seconds'] for row in rows),
            'cuda_stream_span_s_max':max(row['cuda_stream_span_seconds'] for row in rows),
            'footprint_per_rank':{str(row['rank']):row['footprint'] for row in rows},
            'phase_peak_bytes':None,
            'rank_peak_bytes_upper_bound':max(report['initialization_peak_bytes'] if stage=='initialization'
                                             else report['training_step_peak_bytes'] for report in reports)})
    result={'schema':'genrec.issue252.profile/1','status':'PROFILED','formal_training':False,
        'world_size':4,'global_batch':4096,'completed_optimizer_steps':1,'completed_epochs':0,
        'ranks':ranks,'phases':phases,'executed_sources':source_maps[0],
        'raw_reports':{str((DIRECTORY/f'rank_{r}.json').relative_to(ROOT)):sha(DIRECTORY/f'rank_{r}.json') for r in range(4)},
        'executed_source_snapshot':{'path':str(snapshot.relative_to(ROOT)),'sha256':sha(snapshot)},
        'historical_timing':{'path':'results/research/issue252/HISTORICAL_TIMING.json',
                             'sha256':sha(ROOT/'results/research/issue252/HISTORICAL_TIMING.json')},
        'collector':{'path':str(Path(__file__).relative_to(ROOT)),'sha256':sha(__file__)},
        'limits':['Nested inclusive phase spans must not be added.',
                  'One step is not a completed epoch or a throughput credential.',
                  'Only phase-global allocator peaks were recorded; phase peaks are unmeasured, with observed rank-peak upper bounds.',
                  'Raw legacy device operator fields were mislabeled native microseconds; corrected by1e6 here, not by rerunning training.',
                  'Full diagnostic command took1233.49s including profiler collection/serialization; initialization/step scopes exclude profiler teardown.']}
    output=ROOT/'results/research/issue252/BASELINE_PROFILE.json'
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(output.relative_to(ROOT)),'sha256':sha(output),
        'rank_step_seconds':[r['actual_one_step_wall_seconds'] for r in ranks],
        'rank_peak_bytes':[r['training_step_peak_bytes'] for r in ranks]}))


if __name__=='__main__':main()
