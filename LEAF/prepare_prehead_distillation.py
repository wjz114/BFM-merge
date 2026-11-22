"""Freeze one native-prehead distillation experiment and its causal target controls."""
import copy
import json
import time
from pathlib import Path

from d_downstream_qformer_ce import sha256
from prepare_duration_ce_controls import write_new
from prehead_distillation_support import CONDITIONS


def main():
    root = Path('runs/paper_merge_v2')
    template_path = root/'inputs/full_model_distillation_training_plan.json'
    template = json.loads(template_path.read_text())
    previous = json.loads((root/'inputs/preprojection_design.json').read_text())
    source_fit = Path(template['source_fit'])/'run.json'
    fit = json.loads(source_fit.read_text())
    sources = {**template['source_sha256'], **previous['source_sha256']}
    for name, digest in sources.items():assert sha256(Path(name)) == digest, name
    for name in ['prehead_distillation_support.py','cache_prehead_distillation.py',
                 'calibrate_prehead_distillation.py','train_prehead_distillation.py',
                 'check_prehead_ce_replay.py','prepare_prehead_distillation.py',
                 'materialize_prehead_training_plan.py']:
        sources[name] = sha256(Path(name))
    files = [source_fit, template_path, Path(template['initial_checkpoint']),
             Path(template['coverage_audit']), Path(template['schedule']), Path(template['calibration']),
             Path(template['target_cache']), root/'analysis/preprojection_representation_summary.json',
             root/'analysis/duration4800_ce_v2_summary.json']
    inputs = {str(path):sha256(path) for path in files}
    common = {'source_fit':str(source_fit),'initial_checkpoint':template['initial_checkpoint'],
        'coverage_audit':template['coverage_audit'],'schedule':template['schedule'],
        'reference_targets':template['target_cache'], 'expert_sha256':
            {task:sha256(Path(path)) for task,path in fit['experts'].items()},
        'source_sha256':sources,'input_sha256':inputs,'seed':0,'batch_size':128,
        'scope':'Native initial/own-expert2048 prehead features on exactly the existing unique fitting rows plus original512 replay rows. Selected training EEG reads only; no validation inference or optimization. Class means use the fitting multiset. Shuffled targets are a fixed derangement within true class AND fitting multiplicity, preserving the entire fitting target multiset and one target per repeated input; singleton strata remain unchanged and are reported.'}
    design = {**copy.deepcopy(common),'fixed_unix':time.time(),'datasets':fit['datasets'],
        'training_template':str(template_path),'training_template_sha256':sha256(template_path),
        'conditions':list(CONDITIONS),'steps':1600,'training_batch_size':128,'lr':1e-5,
        'eval_every':160,'average_steps':[800,960,1120,1280,1440,1600],
        'target_manifest':str(root/'seed0/prehead_distillation_targets.json'),
        'scales':str(root/'analysis/prehead_distillation_scales.json'),
        'training_plan':str(root/'inputs/prehead_distillation_training_plan.json'),
        'hypothesis':'Own experts retain instance variation before their final projection. Test whether paired native-prehead matching improves the shared model beyond ordinary CE, and whether any benefit depends on sample pairing rather than teacher class means or the class-conditional target distribution.',
        'objective':'BalancedCE + alpha_task * weighted mean over examples of mean2048((h_student-target)^2). Targets: paired own-expert h, fitting-multiset true-class mean h, or class-and-multiplicity-preserving shuffled h. No feature normalization, learned projector, KL mixture or extra inference head.',
        'scale_rule':'At the fixed initial merger, alpha_task is the RMS weighted CE gradient at2048 prehead divided by the RMS paired-feature MSE gradient there on all12800 fitting exposures/task. Compute inFP64 once. Use this SAME alpha for mean and shuffle, with no rematching/tuning; report their initial gradient strengths and alignments. This is not matching parameter gradients or earlier768-output KD scales.',
        'shuffle_rule':'For each task, use numpy default_rng([0,task_index,20261118]); partition unique fitting rows by true class and positive exposure multiplicity. Randomly order each stratum, then cyclically map to the preceding row. Strata of size1 stay unchanged. Zero-exposure replay rows are untouched. Preserve exact target multiset, class means/covariances and repeated-input consistency.',
        'primary_contrasts':['paired minus matched ordinaryCE','paired minus class mean','paired minus class-preserving shuffle'],
        'secondary_contrasts':['mean minus CE','shuffle minus CE','paired versus historical outputKL and final-featureKD (different gradient interface; not a pure layer-location contrast)'],
        'reporting':'Fixed six-state means, all16task effects, macro BAcc, paired signs and leave-one-task-out sensitivity; selected/final states secondary. No task deletion or hyperparameter selection from results. One bank exposed validation, not independent confirmation.',
        'engineering_gate':'Cache input/model/source hashes, exact native head reconstruction, old output cache replay<2e-5 with changed decisions bounded by margins, frozen states exact. FP64 CE/MSE derivatives checked independently by autograd for all targets. Deterministic32-update zero-auxiliary GPU replay must exactly match existing CE states/metrics; one32-update paired-target smoke must have finite nonzero gradients and native export. Deterministic flags apply only to the engineering replay; full scientific trajectories retain native nondeterministic settings.',
        'native_export':'Ordinary single LEAF checkpoint, unchanged architecture, text prototypes and native inference; no runtime teacher, routing, auxiliary head or new parameters.',
        'budget':{'training_updates_per_arm':1600,'training_exposures_per_arm':204800,'scientific_arms':4,
                  'unique_fitting_rows_per_arm':117785,'cache_forward_rows_per_model':118630,
                  'cache_models':2,'engineering_updates':64,
                  'prior_expert_bank_updates':48000,'prior_coefficient_updates':1600},
        'prior_art':['https://arxiv.org/html/2409.20449v1','https://arxiv.org/pdf/1904.05068'],
        'interpretation_limits':'Conventional featureKD plus diagnostic target interventions, no algorithmic novelty claim. Any paired-versus-mean difference includes the removed target variation and changed gradient strength; shuffle preserves target distribution but changes input-target relation. Neither alone proves physiological meaning or independent generalization. Stable material gains would require later replication and a differentiated mechanism/method.',
        'cache_shards':{str(i):{'tasks':fit['datasets'][i::2],'gpu':i,
            'plan':str(root/f'inputs/prehead_distillation_cache_shard{i}_plan.json'),
            'outdir':str(root/f'seed0/prehead_distillation_cache_shard{i}'),
            'log':str(root/f'seed0/logs/prehead_distillation_cache_shard{i}.log')} for i in range(2)},
        'training_arms':{name:{'outdir':str(root/f'seed0/prehead_distillation_{name}'),
            'log':str(root/f'seed0/logs/prehead_distillation_{name}.log'),
            'gpu':i%2,'wave':i//2} for i,name in enumerate(['ce_prehead_paired','balanced_ce','ce_prehead_mean','ce_prehead_shuffled'])}}
    path=root/'inputs/prehead_distillation_design.json'
    write_new(path,design)
    for i,spec in design['cache_shards'].items():
        write_new(Path(spec['plan']),{**copy.deepcopy(common),'design':str(path),
            'design_sha256':sha256(path),'tasks':spec['tasks'],'shard':int(i)})
    print(json.dumps({'design':str(path),'sha256':sha256(path),'budget':design['budget']},indent=2))


if __name__=='__main__':main()
