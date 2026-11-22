"""Verify the changed accumulation loop in its native128 configuration."""
import os
import sys

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
import train_task_mixing_ce as trainer


if __name__ == '__main__':
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    sys.argv[0] = trainer.__file__
    trainer.main()
