"""Deterministic numerical comparison; full scientific runs keep their recipe."""
import argparse
import os
import sys

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainer', choices=['reference','coverage'], required=True)
    args, remaining = parser.parse_known_args()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    if args.trainer == 'reference':
        import train_full_model_control as trainer
        from check_coverage_reference import train_validation_only
        trainer.load_dataset = train_validation_only
    else:
        import train_coverage_control as trainer
    sys.argv = [trainer.__file__, *remaining]
    trainer.main()
