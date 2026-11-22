"""Plot all off-domain RI outcomes, retaining calibration-defined useful pairs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def main():
    root = Path('runs/paper_merge_v2')
    report = json.loads((root/'analysis/ri_adaptation_summary.json').read_text())
    tasks = report['expected_experts']
    assert not report['partial'] and len(tasks) == 16 and len(report['pairs']) == 240
    before, after = np.full((16,16), np.nan), np.full((16,16), np.nan)
    protected = []
    for row in report['pairs']:
        x, y = tasks.index(row['expert']), tasks.index(row['target'])
        values = row['splits']['validation']
        before[y,x] = values['before_gain_vs_base_pp']
        after[y,x] = values['after_gain_vs_base_pp']
        if row['calibration_category_before_adaptation'] == 'helpful':
            protected.append((x,y))
    difference = after-before
    limit = max(10, 10*np.ceil(np.nanmax(np.abs([before,after]))/10))
    change_limit = max(5, 5*np.ceil(np.nanmax(np.abs(difference))/5))
    labels = [t.replace('MI_', '').replace('EMO_', '').replace('_seg4', '') for t in tasks]
    cmap = plt.get_cmap('RdBu').copy()
    cmap.set_bad('#dddddd')
    fig, axes = plt.subplots(1,3,figsize=(17,7),layout='constrained')
    images=[]
    for axis, values, title, bound in zip(axes,[before,after,difference],
            ['Original expert minus base','RI-adapted expert minus base','RI minus original expert'],
            [limit,limit,change_limit]):
        images.append(axis.imshow(values,vmin=-bound,vmax=bound,cmap=cmap))
        for x,y in protected:
            axis.add_patch(Rectangle((x-.48,y-.48),.96,.96,fill=False,edgecolor='black',linewidth=.9))
        axis.set_title(title,fontsize=12)
        axis.set_xticks(range(16),labels,rotation=90,fontsize=7)
        axis.set_yticks(range(16),labels,fontsize=7)
        axis.set_xlabel('Source expert')
        axis.set_ylabel('Target domain')
    fig.colorbar(images[0],ax=axes[:2],shrink=.65,label='Validation BAcc difference (percentage points)')
    fig.colorbar(images[2],ax=axes[2],shrink=.65,label='Adaptation change (percentage points)')
    fig.suptitle('Cross-domain effects before and after RI adaptation',fontsize=15)
    fig.supxlabel('Fixed validation probes: up to 512 examples per domain. Black outlines: helpful on pre-adaptation calibration.\n'
                  'Grey diagonal omitted. Development diagnostics; no independent or merged-model performance claim.',fontsize=9)
    folder=root/'analysis/figures';folder.mkdir(exist_ok=True)
    for suffix in ['png','pdf']:
        fig.savefig(folder/f'ri_transfer_matrix.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    main()
