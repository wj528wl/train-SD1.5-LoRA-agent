import argparse
import os
import sys

if __package__ in {None, ""}:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from train.train_lora import add_training_arguments, build_config_from_args, train
else:
    from .train_lora import add_training_arguments, build_config_from_args, train


DEFAULTS = {
    "config": None,
    "dreambooth": False,
    "model_path": "./models/anything-v5",
    "data_dir": "./data",
    "resolution": 512,
    "center_crop": True,
    "batch_size": 4,
    "num_epochs": 30,
    "learning_rate": 1e-4,
    "lr_scheduler": "constant",
    "lr_warmup_steps": 0,
    "optimizer": "adamw",
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_weight_decay": 1e-2,
    "adam_epsilon": 1e-8,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "output_dir": "./outputs/lora_full",
    "save_steps": 500,
    "save_total_limit": 5,
    "logging_steps": 10,
    "device": "cuda",
    "mixed_precision": "fp16",
    "seed": 42,
    "num_workers": 0,
    "lora_rank": 16,
    "lora_alpha": 16.0,
    "instance_dir": "./data/instance",
    "class_dir": "",
    "instance_prompt": "a photo of sks person",
    "class_prompt": "a photo of person",
    "prior_loss_weight": 1.0,
}

BOOLEAN_FLAGS = {
    "dreambooth": "--dreambooth",
}

VALUE_FLAGS = {
    "config": "--config",
    "model_path": "--model_path",
    "data_dir": "--data_dir",
    "resolution": "--resolution",
    "center_crop": "--center_crop",
    "batch_size": "--batch_size",
    "num_epochs": "--num_epochs",
    "learning_rate": "--learning_rate",
    "lr_scheduler": "--lr_scheduler",
    "lr_warmup_steps": "--lr_warmup_steps",
    "optimizer": "--optimizer",
    "adam_beta1": "--adam_beta1",
    "adam_beta2": "--adam_beta2",
    "adam_weight_decay": "--adam_weight_decay",
    "adam_epsilon": "--adam_epsilon",
    "gradient_accumulation_steps": "--gradient_accumulation_steps",
    "max_grad_norm": "--max_grad_norm",
    "output_dir": "--output_dir",
    "save_steps": "--save_steps",
    "save_total_limit": "--save_total_limit",
    "logging_steps": "--logging_steps",
    "device": "--device",
    "mixed_precision": "--mixed_precision",
    "seed": "--seed",
    "num_workers": "--num_workers",
    "lora_rank": "--lora_rank",
    "lora_alpha": "--lora_alpha",
    "instance_dir": "--instance_dir",
    "class_dir": "--class_dir",
    "instance_prompt": "--instance_prompt",
    "class_prompt": "--class_prompt",
    "prior_loss_weight": "--prior_loss_weight",
}


def build_parser():
    parser = argparse.ArgumentParser(description="Train SD1.5 LoRA weights.")
    return add_training_arguments(parser)


def _cli_mode_requested(argv=None):
    return argv is not None and len(argv) > 0


def _build_args_from_defaults(parser):
    cli_args = []
    use_dreambooth = bool(DEFAULTS.get("dreambooth"))

    for key, flag in BOOLEAN_FLAGS.items():
        if DEFAULTS.get(key):
            cli_args.append(flag)

    for key, flag in VALUE_FLAGS.items():
        value = DEFAULTS.get(key)
        if value is None:
            continue
        if not use_dreambooth and key in {"instance_dir", "class_dir", "instance_prompt", "class_prompt", "prior_loss_weight"}:
            continue
        cli_args.extend([flag, str(value)])

    return parser.parse_args(cli_args)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()

    if _cli_mode_requested(argv):
        args = parser.parse_args(argv)
    else:
        args = _build_args_from_defaults(parser)

    config = build_config_from_args(args)
    train(config)


if __name__ == "__main__":
    main()
