"""Compare fixed merger and own-expert fit on reused and replacement rows."""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from adapt_experts_ri import configure
from aligned_support import array_hash
from check_merge_export_margins import compare
from d_downstream_qformer_ce import ROOT, PrototypeClassifier, classification_metrics, load_leaf, make_loader, prepare_text, sha256
from diagnose_sensor_subspaces import read_rows, row_map
from leaf_datasets.shared import DATA_FOLDER
from load_config import build_model_config, load_yaml


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for filename, expected in {**plan['source_sha256'], **plan['input_sha256']}.items():
        assert sha256(Path(filename)) == expected, filename
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    device = configure(args.gpu, 0)
    training_plan = json.loads(Path(plan['training_plan']).read_text())
    source = json.loads((Path(training_plan['source_fit'])/'run.json').read_text())
    previous = json.loads(Path(plan['initial_reference']).read_text())
    data_audit = json.loads(Path(training_plan['fresh_data_audit']).read_text())
    coverage = json.loads(Path(training_plan['coverage_audit']).read_text())
    old = np.load(training_plan['calibration'])
    fresh = np.load(training_plan['training_indices'])
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, Path(training_plan['initial_checkpoint'])).to(device).eval().requires_grad_(False)
    initial = {name: value.detach().cpu().clone() for name, value in leaf.state_dict().items()}
    assert sha256(Path(training_plan['initial_checkpoint'])) == previous['checkpoint_sha256']['initial']
    evaluations, cache_checks, input_records = {}, {}, {}
    for task in source['datasets']:
        with h5py.File(DATA_FOLDER/f'{task}.h5', 'r', locking=False) as h:
            rows, y = row_map(h, task, 'calibration')
            y = np.asarray(y, dtype=np.int64)
            assert array_hash(y) == coverage['audits'][task]['train_labels_hash']
            selections = {name: [rows[int(i)] for i in ids] for name, ids in [('old', old[task]), ('fresh', fresh[task])]}
            keys = dict.fromkeys(row[0] for selection in selections.values() for row in selection)
            memory = {key: h[key][:] for key in keys}
        x_old = read_rows(memory, selections['old'])
        x_fresh = read_rows(memory, selections['fresh'])
        del memory
        assert array_hash(x_old) == coverage['audits'][task]['calibration_eeg_hash']
        assert np.array_equal(y[old[task]], y[fresh[task]])
        input_records[task] = {'old_eeg_hash': array_hash(x_old), 'fresh_eeg_hash': array_hash(x_fresh),
                               'overlap': data_audit['datasets'][task]['observed_overlap']}
        labels, _, instruction, prototypes = prepare_text(config, task)
        wrapper = PrototypeClassifier(leaf, instruction, prototypes).to(device).eval()
        x = np.concatenate([x_old, x_fresh])
        truth = torch.from_numpy(np.concatenate([y[old[task]], y[fresh[task]]]))
        del x_old, x_fresh, y
        loader = make_loader(x, truth.numpy(), config, len(labels), 128, False)
        cache = torch.load(Path(training_plan['teacher_cache'])/f'{task}.pt', map_location='cpu', weights_only=True)
        assert sha256(Path(source['experts'][task])) == cache['metadata']['expert_sha256']
        logits, metrics = {}, {}
        for name in ['initial_merger', 'own_expert']:
            state = initial if name == 'initial_merger' else torch.load(source['experts'][task], map_location='cpu', weights_only=True)
            leaf.load_state_dict(state, strict=True)
            values = torch.cat([wrapper(xx.to(device)).cpu() for xx, _ in loader])
            logits[name] = values
            metrics[name] = {pool: classification_metrics(values[sl], truth[sl])
                             for pool, sl in [('old', slice(0, 512)), ('fresh', slice(512, None))]}
        assert metrics['initial_merger']['old']['BAcc'] == previous['evaluations']['initial'][task]['calibration']['BAcc']
        cache_checks[task] = compare(cache['own_expert_logits'][:512], logits['own_expert'][:512], truth[:512])
        assert cache_checks[task]['max_abs_logit_error'] <= 1e-5
        assert all(row['margin_within_twice_error'] for row in cache_checks[task]['changed_rows'])
        evaluations[task] = metrics
        torch.save({'truth': truth, 'logits': logits}, args.outdir/f'{task}.pt')
        (args.outdir/'progress.json').write_text(json.dumps(evaluations, indent=2)+'\n')
        print(task, {model: {pool: value['BAcc'] for pool, value in pools.items()}
                     for model, pools in metrics.items()}, flush=True)
    summaries = {}
    for label, tasks in [('all16', source['datasets']), ('fully_disjoint15', data_audit['fully_disjoint_tasks'])]:
        summaries[label] = {
            model: {pool: {metric: float(np.mean([evaluations[t][model][pool][metric] for t in tasks]))
                           for metric in ['BAcc', 'Loss', 'Auroc']}
                    for pool in ['old', 'fresh']}
            for model in ['initial_merger', 'own_expert']
        }
    result = {'plan': plan, 'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
              'evaluations': evaluations, 'summaries': summaries, 'expert_cache_checks': cache_checks,
              'input_records': input_records, 'initial_calibration_bacc_replay_exact': True,
              'seconds': time.time()-started, 'test_data_read': False, 'test_inference_performed': False,
              'validation_inference_performed': False, 'fitting_performed': False,
              'scope': 'Frozen initial merger versus each own expert on old and minimally overlapping replacement512 rows. Every row is from original expert training. Class counts and label positions match. Expert old/fresh comparison is a descriptive sample-difficulty control, not a causal adjustment or independent evaluation. Training intervention remains separate.'}
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Completed', json.dumps(summaries['all16']), flush=True)


if __name__ == '__main__':
    main()
