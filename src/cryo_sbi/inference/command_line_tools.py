import argparse
from typing import Union, Optional
from cryo_sbi.inference.train_npe_model import (
    npe_train_no_saving
)

from cryo_sbi.inference.train_nle_model_with_finetuning import (
    nle_train_no_saving_with_finetuning
)


def cl_npe_train_no_saving():
    cl_parser = argparse.ArgumentParser()

    cl_parser.add_argument(
        "--image_config_file", action="store", type=str, required=True
    )
    cl_parser.add_argument(
        "--train_config_file", action="store", type=str, required=True
    )
    cl_parser.add_argument("--epochs", action="store", type=int, required=True)
    cl_parser.add_argument("--estimator_file", action="store", type=str, required=True)
    cl_parser.add_argument("--loss_file", action="store", type=str, required=True)
    cl_parser.add_argument(
        "--train_from_checkpoint",
        action="store",
        type=bool,
        nargs="?",
        required=False,
        const=True,
        default=False,
    )
    cl_parser.add_argument(
        "--state_dict_file", action="store", type=str, required=False, default=False
    )
    cl_parser.add_argument(
        "--n_workers", action="store", type=int, required=False, default=1
    )
    cl_parser.add_argument(
        "--train_device", action="store", type=str, required=False, default="cpu"
    )
    cl_parser.add_argument(
        "--saving_freq", action="store", type=int, required=False, default=20
    )
    cl_parser.add_argument(
        "--simulation_batch_size",
        action="store",
        type=int,
        required=False,
        default=1024,
    )

    args = cl_parser.parse_args()

    npe_train_no_saving(
        image_config=args.image_config_file,
        train_config=args.train_config_file,
        epochs=args.epochs,
        estimator_file=args.estimator_file,
        loss_file=args.loss_file,
        train_from_checkpoint=args.train_from_checkpoint,
        model_state_dict=args.state_dict_file,
        n_workers=args.n_workers,
        device=args.train_device,
        saving_frequency=args.saving_freq,
        simulation_batch_size=args.simulation_batch_size,
    )


def cl_nle_train_no_saving_with_finetuning():
    cl_parser = argparse.ArgumentParser()

    cl_parser.add_argument(
        "--image_config_file", action="store", type=str, required=True
    )
    cl_parser.add_argument(
        "--train_config_file", action="store", type=str, required=True
    )
    cl_parser.add_argument("--epochs", action="store", type=int, required=True)
    cl_parser.add_argument("--estimator_file", action="store", type=str, required=True)
    cl_parser.add_argument("--loss_file", action="store", type=str, required=True)
    cl_parser.add_argument("--log_file", action="store", type=str, required=True)
    cl_parser.add_argument(
        "--train_from_checkpoint",
        action="store",
        type=bool,
        nargs="?",
        required=False,
        const=True,
        default=False,
    )
    cl_parser.add_argument(
        "--state_dict_file", action="store", type=str, required=False, default=False
    )
    cl_parser.add_argument(
        "--n_workers", action="store", type=int, required=False, default=1
    )
    cl_parser.add_argument(
        "--train_device", action="store", type=str, required=False, default="cuda"
    )
    cl_parser.add_argument(
        "--saving_freq", action="store", type=int, required=False, default=10
    )
    cl_parser.add_argument(
        "--simulation_batch_size",
        action="store",
        type=int,
        required=False,
        default=2048,
    )
    cl_parser.add_argument(
        "--n_batches_per_epoch",
        type=int,
        default=100,
        help="Number of simulation batches to generate per epoch (default: 100)",
    )
    cl_parser.add_argument(
        "--pretrained_embedding_path",
        action="store",
        type=str,
        required=False,
        default=None,
        help="Path to pretrained encoder weights.",
    )
    cl_parser.add_argument(
        "--freeze_embedding",
        action="store",
        type=bool,
        required=False,
        default=False,
    )
    cl_parser.add_argument(
        "--use_differential_lr",
        action="store",
        type=bool,
        required=False,
        default=False,
    )
    cl_parser.add_argument(
        "--embedding_lr_factor",
        action="store",
        type=float,
        required=False,
        default=0.01,
    )
    cl_parser.add_argument(
        "--real_data_mrc",
        action="store",
        type=str,
        required=False,
        default=None,
        help="Path to .mrc stack of real images.",
    )
    cl_parser.add_argument(
        "--n_real_val",
        action="store",
        type=int,
        required=False,
        default=3000,
        help="Number of real images to hold in GPU memory for validation (default: 3000).",
    )
    cl_parser.add_argument(
        "--fraction_finetune_epochs",
        action="store",
        type=float,
        required=False,
        default=0.0,
        help="Fraction of final training epochs allocated to real-data fine-tuning (default: 0.0, disabled).",
    )
    cl_parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Enable exponential moving average of model weights.",
    )
    cl_parser.add_argument(
        "--ema_decay",
        action="store",
        type=float,
        required=False,
        default=0.999,
        help="EMA decay coefficient (default: 0.999).",
    )
    cl_parser.add_argument(
        "--ema_start_step",
        action="store",
        type=int,
        required=False,
        default=0,
        help="Number of optimizer steps before EMA averaging starts (default: 0).",
    )
    cl_parser.add_argument(
        "--ema_save_both",
        action="store_true",
        help="If set, also save a non-EMA checkpoint alongside the EMA checkpoint.",
    )

    args = cl_parser.parse_args()

    nle_train_no_saving_with_finetuning(
        image_config=args.image_config_file,
        train_config=args.train_config_file,
        epochs=args.epochs,
        estimator_file=args.estimator_file,
        loss_file=args.loss_file,
        log_file=args.log_file,
        train_from_checkpoint=args.train_from_checkpoint,
        model_state_dict=args.state_dict_file,
        n_workers=args.n_workers,
        device=args.train_device,
        saving_frequency=args.saving_freq,
        simulation_batch_size=args.simulation_batch_size,
        n_batches_per_epoch=args.n_batches_per_epoch,
        pretrained_embedding_path=args.pretrained_embedding_path,
        freeze_embedding=args.freeze_embedding,
        use_differential_lr=args.use_differential_lr,
        embedding_lr_factor=args.embedding_lr_factor,
        real_data_mrc=args.real_data_mrc,
        n_real_val=args.n_real_val,
        fraction_finetune_epochs=args.fraction_finetune_epochs,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        ema_start_step=args.ema_start_step,
        ema_save_both=args.ema_save_both,
    )
