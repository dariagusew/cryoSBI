import argparse
from typing import Union, Optional
from cryo_sbi.inference.train_npe_model import (
    npe_train_no_saving
)
from cryo_sbi.inference.train_nle_model import (
    nle_train_no_saving
)

from cryo_sbi.inference.train_nle_model_with_validation import (
    nle_train_no_saving_with_validation
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


def cl_nle_train_no_saving_with_validation():
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
        "--pretrained_embedding_path",
        action="store",
        type=str,
        required=True,
        default=None,
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

    cl_parser.add_argument("--validation_mrc_path", action="store", type=str, required=False, default=None)
 

    args = cl_parser.parse_args()

    nle_train_no_saving_with_validation(
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
        pretrained_embedding_path=args.pretrained_embedding_path,
        freeze_embedding=args.freeze_embedding,
        use_differential_lr=args.use_differential_lr,
        embedding_lr_factor=args.embedding_lr_factor,
        validation_mrc_path=args.validation_mrc_path
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
        "--pretrained_embedding_path",
        action="store",
        type=str,
        required=True,
        default=None,
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
        "--real_data_fraction",
        action="store",
        type=float,
        required=False,
        default=0.0,
    )

    cl_parser.add_argument("--validation_mrc_path", action="store", type=str, required=False, default=None)

    cl_parser.add_argument('--sample_indices', action='store_true')

    args = cl_parser.parse_args()

    nle_train_no_saving_with_finetuning(
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
        pretrained_embedding_path=args.pretrained_embedding_path,
        freeze_embedding=args.freeze_embedding,
        use_differential_lr=args.use_differential_lr,
        embedding_lr_factor=args.embedding_lr_factor,
        validation_mrc_path=args.validation_mrc_path,
        real_data_finetune_fraction=args.real_data_fraction,
        sample_indices=args.sample_indices
    )


def cl_nle_train_no_saving():
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
        "--pretrained_embedding_path",
        action="store",
        type=str,
        required=True,
        default=None,
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

    args = cl_parser.parse_args()

    nle_train_no_saving(
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
        pretrained_embedding_path=args.pretrained_embedding_path,
        freeze_embedding=args.freeze_embedding,
        use_differential_lr=args.use_differential_lr,
        embedding_lr_factor=args.embedding_lr_factor
    )
