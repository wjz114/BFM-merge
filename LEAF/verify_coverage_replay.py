"""Numerically compare the frozen and new original512 training paths."""
import argparse
import json
from pathlib import Path

import torch

from d_downstream_qformer_ce import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default='coverage_smoke_reference')
    parser.add_argument('--candidate', default='coverage_smoke_original512')
    parser.add_argument('--output', default='full_control_coverage_replay_verification.json')
    args = parser.parse_args()
    root = Path('runs/paper_merge_v2')
    old, new = [root/'seed0'/name for name in [args.reference, args.candidate]]
    a, b = [json.loads((p/'run.json').read_text()) for p in [old, new]]
    assert a['engineering_smoke'] and b['engineering_smoke'] and a['updates'] == b['updates'] == 32
    assert a['datasets'] == b['datasets'] and a['schedule_sha256'] == b['schedule_sha256']
    assert a['fixed_tensors_exact'] and b['fixed_tensors_exact']
    records = {}
    for name in ['step_16.ckpt','step_32.ckpt','selected.ckpt','final.ckpt','merged.ckpt']:
        sa, sb = [torch.load(p/name, map_location='cpu', weights_only=True) for p in [old,new]]
        assert sa.keys() == sb.keys()
        differences = {}
        for key in sa:
            if not torch.equal(sa[key], sb[key]):
                assert torch.isfinite(sa[key]).all() and torch.isfinite(sb[key]).all(), key
                differences[key] = float((sa[key].double()-sb[key].double()).abs().max())
        records[name] = {'all_tensors_bit_exact':not differences,'max_parameter_difference':max(differences.values(),default=0.),
                         'different_tensors':differences,'old_sha256':sha256(old/name),'new_sha256':sha256(new/name)}
        assert not differences, (name, differences)
    histories = [json.loads((p/'history.json').read_text()) for p in [old,new]]
    assert histories[0] == histories[1]
    assert a['average_validation'] == b['average_validation']
    assert a['average_calibration'] == b['average_calibration']
    for task in a['datasets']:
        for key in ['eeg_hash','class_counts','teacher_hash','teacher_expert_sha256']:
            assert a['dataset_audits'][task][key] == b['dataset_audits'][task][key], (task,key)
        assert a['export_checks'][task]['max_abs_logit_error'] <= 1e-5
        assert b['export_checks'][task]['max_abs_logit_error'] <= 1e-5
    result = {'status':'passed','checkpoints':records,'histories_exact':True,'all_validation_and_calibration_metrics_exact':True,
              'old_run_sha256':sha256(old/'run.json'),'new_run_sha256':sha256(new/'run.json'),
              'code_sha256':sha256(Path(__file__)),'scope':'32updates across all16 tasks with the exact original512 schedule; five exported states are bit-exact, and every history/average metric agrees. Frozen reference training code uses only an I/O adapter that omits discarded official-test arrays. This verifies implementation equivalence, not scientific performance.'}
    (root/'analysis'/args.output).write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: five exported states, all histories and all averaged metrics are bit-exact')


if __name__ == '__main__':
    main()
