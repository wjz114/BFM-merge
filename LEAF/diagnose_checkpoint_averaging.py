"""Compare a fixed checkpoint weight mean with its fixed logit ensemble."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from adapt_experts_ri import configure
from aligned_support import array_hash
from d_downstream_qformer_ce import ROOT, PrototypeClassifier, classification_metrics, load_leaf, make_loader, prepare_text, sha256
from diagnose_sensor_projection import calibration_and_validation
from load_config import build_model_config, load_yaml
from merge_poc import UNUSED_PREFIXES


def compare_predictions(weight, ensemble, members, truth):
    weight, ensemble, members = weight.double(), ensemble.double(), members.double()
    wc = weight-weight.mean(-1, keepdim=True)
    ec = ensemble-ensemble.mean(-1, keepdim=True)
    mc = members-members.mean(-1, keepdim=True)
    residual = wc-ec
    variance = (mc-ec).square().mean()
    wp, ep = weight.argmax(-1), ensemble.argmax(-1)
    return {
        'n': len(truth),
        'prediction_agreement': float((wp == ep).double().mean()),
        'disagreement_count': int((wp != ep).sum()),
        'weight_correct_ensemble_wrong': int(((wp == truth) & (ep != truth)).sum()),
        'ensemble_correct_weight_wrong': int(((ep == truth) & (wp != truth)).sum()),
        'centered_logit_residual_rms': float(residual.square().mean().sqrt()),
        'centered_ensemble_logit_rms': float(ec.square().mean().sqrt()),
        'relative_centered_residual_norm': float(residual.norm()/ec.norm()),
        'centered_snapshot_dispersion_rms': float(variance.sqrt()),
        'residual_rms_over_snapshot_dispersion': float(residual.square().mean().sqrt()/variance.sqrt()),
        'mean_snapshot_prediction_agreement_with_ensemble': float((members.argmax(-1) == ep).double().mean()),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for path, expected in plan['source_sha256'].items():
        assert sha256(Path(path)) == expected, path
    source_dir = Path(plan['source'])
    for path, expected in plan['input_sha256'].items():
        assert sha256(Path(path)) == expected, path
    source = json.loads((source_dir/'run.json').read_text())
    history = {row['step']: row for row in json.loads((source_dir/'history.json').read_text())}
    assert source['average_steps'] == plan['snapshots']
    tasks = source['datasets']
    args.outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    device = configure(args.gpu, 0)
    states = {'original': torch.load(source['plan']['initial_checkpoint'], map_location='cpu', weights_only=True)}
    members = [f'step_{step}' for step in plan['snapshots']]
    for name in members:
        states[name] = torch.load(source_dir/f'{name}.ckpt', map_location='cpu', weights_only=True)
    states['weight_mean'] = torch.load(source_dir/'merged.ckpt', map_location='cpu', weights_only=True)
    config = build_model_config(load_yaml(ROOT/'configs/LEAF_mpnet.yaml'))
    leaf = load_leaf(config, source_dir/'merged.ckpt').to(device).eval().requires_grad_(False)
    active = {name for name, _ in leaf.named_parameters() if not name.startswith(UNUSED_PREFIXES)}
    for name, value in states['weight_mean'].items():
        expected = (torch.stack([states[m][name].double() for m in members]).mean(0).float()
                    if name in active else states['original'][name])
        assert torch.equal(value, expected), name
    indices = np.load(source['plan']['calibration'])
    evaluations = {name: {} for name in [*states, 'logit_mean']}
    comparisons = {}
    for task in tasks:
        labels, _, instruction, prototypes = prepare_text(config, task)
        wrapper = PrototypeClassifier(leaf, instruction, prototypes).to(device).eval()
        cx, cy, vx, vy = calibration_and_validation(task, indices[task])
        assert array_hash(cx) == source['dataset_audits'][task]['eeg_hash']
        assert len(vy) == source['dataset_audits'][task]['validation_size']
        n = len(cy)
        x, y = np.concatenate([cx, vx]), np.concatenate([cy, vy])
        del cx, cy, vx, vy
        truth = torch.from_numpy(y)
        loader = make_loader(x, y, config, len(labels), plan['batch_size'], False)
        logits = {}
        for name, state in states.items():
            leaf.load_state_dict(state, strict=True)
            logits[name] = torch.cat([wrapper(xx.to(device)).cpu() for xx, _ in loader])
            assert torch.isfinite(logits[name]).all()
        member_logits = torch.stack([logits[m] for m in members])
        logits['logit_mean'] = member_logits.double().mean(0).float()
        comparisons[task] = {}
        for name in logits:
            evaluations[name][task] = {}
        for split, ids in [('calibration', slice(0, n)), ('validation', slice(n, None))]:
            for name, values in logits.items():
                evaluations[name][task][split] = classification_metrics(values[ids], truth[ids])
            comparisons[task][split] = compare_predictions(logits['weight_mean'][ids], logits['logit_mean'][ids], member_logits[:, ids], truth[ids])
        # Full validation replay verifies each individual model as well as both endpoints.
        references = {'original': source['initial_validation']['validation'], 'weight_mean': source['average_validation']}
        references.update({f'step_{step}': history[step]['validation'] for step in plan['snapshots']})
        for name, ref in references.items():
            assert abs(evaluations[name][task]['validation']['BAcc']-ref[task]['BAcc']) < 1e-12, (task, name)
        assert abs(evaluations['weight_mean'][task]['calibration']['BAcc']-source['average_calibration'][task]['BAcc']) < 1e-12
        torch.save({'truth': truth, 'n_calibration': n, 'logits': logits}, args.outdir/f'{task}_logits.pt')
        (args.outdir/'progress.json').write_text(json.dumps({'evaluations': evaluations, 'comparisons': comparisons}, indent=2)+'\n')
        print(task, {name: evaluations[name][task]['validation']['BAcc'] for name in ['original', 'weight_mean', 'logit_mean']}, flush=True)
    summaries, contrasts = {}, {}
    for split in ['calibration', 'validation']:
        summaries[split] = {name: float(np.mean([evaluations[name][t][split]['BAcc'] for t in tasks])) for name in evaluations}
        summaries[split]['mean_individual_snapshot_bacc'] = float(np.mean([summaries[split][m] for m in members]))
        summaries[split]['mean_prediction_agreement_weight_vs_logit'] = float(np.mean([comparisons[t][split]['prediction_agreement'] for t in tasks]))
        contrasts[split] = {}
        for treatment, control in [('weight_mean', 'original'), ('logit_mean', 'original'), ('weight_mean', 'logit_mean')]:
            deltas = {t: 100*(evaluations[treatment][t][split]['BAcc']-evaluations[control][t][split]['BAcc']) for t in tasks}
            contrasts[split][treatment+'_vs_'+control] = {'macro_gain_pp': float(np.mean(list(deltas.values()))), 'per_task_gain_pp': deltas,
                'improved': sum(v>0 for v in deltas.values()), 'declined': sum(v<0 for v in deltas.values()),
                'leave_one_task_out_gain_pp': {t: float(np.mean([v for k,v in deltas.items() if k != t])) for t in tasks}}
    result = {'plan': plan, 'plan_sha256': sha256(args.plan), 'code_sha256': sha256(Path(__file__)),
              'summaries': summaries, 'contrasts': contrasts, 'evaluations': evaluations, 'comparisons': comparisons,
              'weight_mean_exact_replay': True, 'all_validation_bacc_exact_replay': True,
              'seconds': time.time()-started, 'test_data_read': False, 'test_inference_performed': False,
              'scope': 'Fixed six-checkpoint prediction ensemble is diagnostic only. Weight mean is a single existing ordinary checkpoint. All16 exposed validation tasks and original calibration; no new fitting, checkpoint selection, or independent confirmation. Agreement alone does not establish flatness or novelty.'}
    (args.outdir/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Completed', summaries['validation'], flush=True)


if __name__ == '__main__':
    main()
