"""Add completed cache hashes to the prespecified prehead training plan; no choices."""
import copy
import json
from pathlib import Path

from d_downstream_qformer_ce import sha256
from prepare_duration_ce_controls import write_new


def main():
    path=Path('runs/paper_merge_v2/inputs/prehead_distillation_design.json')
    d=json.loads(path.read_text())
    for name,digest in {**d['source_sha256'],**d['input_sha256']}.items():
        assert sha256(Path(name))==digest,name
    p=json.loads(Path(d['training_template']).read_text())
    targets=json.loads(Path(d['target_manifest']).read_text())
    scales=json.loads(Path(d['scales']).read_text())
    assert targets['design_sha256']==scales['design_sha256']==sha256(path)
    assert scales['target_manifest_sha256']==sha256(Path(d['target_manifest']))
    for name in ['geometry_basis','geometry_basis_sha256','engineering_verification',
                 'engineering_verification_sha256','design_path','design_sha256']:
        p.pop(name,None)
    p.update(source_sha256=copy.deepcopy(d['source_sha256']), design_path=str(path),
        design_sha256=sha256(path), conditions=d['conditions'],target_cache=d['target_manifest'],
        target_cache_sha256=sha256(Path(d['target_manifest'])),scales=d['scales'],
        scales_sha256=sha256(Path(d['scales'])),
        scope='Fixed same-data1600-update ordinaryCE/paired-prehead/class-mean/class-and-multiplicity-shuffled target comparison. One initial2048-prehead gradient scale calibrated on paired targets and shared by target controls; no parameter-gradient matching or adaptive tuning. Native numerical settings and ordinary LEAF export, same six-state mean. Single-bank exposed validation; no novelty, independent-confirmation or end-to-end efficiency claim.')
    for key in ['steps','eval_every','average_steps','lr']:assert p[key]==d[key]
    assert p['batch_size']==d['training_batch_size']
    write_new(Path(d['training_plan']),p)
    print('Materialized',d['training_plan'],sha256(Path(d['training_plan'])))


if __name__=='__main__':main()
