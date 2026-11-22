"""One native Adam history per scheduled task, acting on one shared parameter set."""
from collections import Counter
import torch


class TaskStateAdam:
    def __init__(self,parameters,task_order,n_tasks,**kwargs):
        self.parameters=list(parameters)
        self.task_order=[int(t) for t in task_order];self.n_tasks=n_tasks;self.steps=0
        assert self.parameters and all(0<=t<n_tasks for t in self.task_order)
        self.optimizers=[torch.optim.Adam(self.parameters,**kwargs) for _ in range(n_tasks)]
        self.param_groups=self.optimizers[0].param_groups
        self.visits=Counter()

    def zero_grad(self,set_to_none=True):
        # Every optimizer references the same parameters; one native clear suffices.
        self.optimizers[0].zero_grad(set_to_none=set_to_none)

    def step(self):
        assert self.steps<len(self.task_order)
        task=self.task_order[self.steps]
        result=self.optimizers[task].step()
        self.visits[task]+=1;self.steps+=1
        return result

    def audit(self):
        assert dict(self.visits)==dict(Counter(self.task_order[:self.steps]))
        pointers=[];records={};moments=0
        for task,opt in enumerate(self.optimizers):
            assert all(a is b for a,b in zip(opt.param_groups[0]['params'],self.parameters))
            counters=[];states=0;energy_m=0.;energy_v=0.
            for p,s in opt.state.items():
                counters.append(int(s['step'].item()));states+=1
                for key in ['exp_avg','exp_avg_sq']:
                    value=s[key];pointers.append(value.data_ptr());moments+=value.numel()
                    assert torch.isfinite(value).all()
                energy_m+=float(s['exp_avg'].double().square().sum())
                energy_v+=float(s['exp_avg_sq'].double().square().sum())
            assert all(c==self.visits[task] for c in counters)
            records[str(task)]={'visits':self.visits[task],'initialized_parameter_states':states,
                                'first_moment_l2':energy_m**.5,'second_moment_l2':energy_v**.5,
                                'bias_correction_counter_values':sorted(set(counters))}
        assert len(pointers)==len(set(pointers))
        return {'updates':self.steps,'tasks':records,'moment_tensor_elements':moments,
                'moment_storages_disjoint':True,'all_optimizers_share_the_same_parameters':True,
                'unvisited_task_states_do_not_decay':True,
                'semantics':'Only the scheduled task native Adam advances; its bias-correction clock counts task visits. All other moment states are held fixed. One shared model is updated after every batch.'}


def engineering_checks(device):
    torch.manual_seed(20261117)
    shapes=[(3,4),(3,),(16,32)]
    a=[torch.nn.Parameter(torch.randn(shape,device=device)) for shape in shapes]
    b=[torch.nn.Parameter(p.detach().clone()) for p in a]
    native=torch.optim.Adam(a,lr=1e-5)
    one=TaskStateAdam(b,[0]*20,1,lr=1e-5)
    for step in range(20):
        native.zero_grad(set_to_none=True);one.zero_grad(set_to_none=True)
        for p,q in zip(a,b):p.grad=torch.randn_like(p);q.grad=p.grad.clone()
        native.step();one.step()
        assert all(torch.equal(p,q) for p,q in zip(a,b))
        for p,q in zip(a,b):
            for key,v in native.state[p].items():assert torch.equal(v,one.optimizers[0].state[q][key])
    shared=torch.nn.Parameter(torch.tensor([1.,-2.],device=device))
    split=TaskStateAdam([shared],[0,1,1,0],2,lr=1e-5)
    shared.grad=torch.tensor([1.,-1.],device=device);split.step()
    saved={k:v.clone() for k,v in split.optimizers[0].state[shared].items()}
    assert not split.optimizers[1].state
    for g in [[2.,3.],[-2.,1.]]:
        split.zero_grad(set_to_none=True);assert shared.grad is None
        shared.grad=torch.tensor(g,device=device);split.step()
        assert all(torch.equal(v,split.optimizers[0].state[shared][k]) for k,v in saved.items())
    shared.grad=torch.tensor([.5,2.],device=device);split.step();audit=split.audit()
    assert audit['tasks']['0']['visits']==audit['tasks']['1']['visits']==2
    return {'device':str(device),'single_task_native_adam_exact_all_parameters_and_states_20_steps':True,
            'inactive_task_state_unchanged_during_other_task_updates':True,'shared_parameter_clear_native':True,
            'multi_task_audit':audit,'synthetic_updates_per_optimizer':{'native_reference':20,'single_task':20,'two_task':4},
            'new_eeg_forwards':0,'native_leaf_updates':0}
