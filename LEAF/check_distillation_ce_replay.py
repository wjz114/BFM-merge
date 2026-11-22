"""Run the distillation worker with zero auxiliary under the existing deterministic replay settings."""
import os
import sys

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
import train_distillation_control as trainer


if __name__ == '__main__':
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    sys.argv[0] = trainer.__file__
    trainer.main()
