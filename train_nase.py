#!/usr/bin/env python
"""
Training script for NASE-style model with CFG.

Usage:
    # Train NASE-style model with NC loss and CFG
    python train_nase.py --base_dir ./data/voicebank-demand \
        --backbone ncsnpp_nase \
        --noise_encoder_type simple \
        --use_nc_loss \
        --nc_loss_weight 0.3 \
        --p_uncond 0.2 \
        --gpus 4

    # Train without NC loss (ablation)
    python train_nase.py --base_dir ./data/voicebank-demand \
        --backbone ncsnpp_nase \
        --p_uncond 0.2 \
        --gpus 4

    # Train without CFG (NASE baseline)
    python train_nase.py --base_dir ./data/voicebank-demand \
        --backbone ncsnpp_nase \
        --use_nc_loss \
        --nc_loss_weight 0.3 \
        --p_uncond 0.0 \
        --gpus 4
"""

import argparse
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from sgmse.backbones import BackboneRegistry
from sgmse.sdes import SDERegistry
from sgmse.model_nase import NASEScoreModel
from sgmse.data_module import SpecsDataModule


def get_argparse_groups(parser):
    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(parser.parse_args([]), a.dest, None) for a in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_dict)
    return groups


def main():
    # Base arguments
    parser = ArgumentParser()
    parser.add_argument("--backbone", type=str, default="ncsnpp_nase",
                        choices=BackboneRegistry.get_all_names(),
                        help="Backbone network architecture")
    parser.add_argument("--sde", type=str, default="ouve",
                        choices=SDERegistry.get_all_names(),
                        help="SDE for diffusion")
    parser.add_argument("--nolog", action='store_true',
                        help="Disable W&B logging (use CSV logger)")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs to use")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--save_ckpt_interval", type=int, default=50000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--max_steps", type=int, default=500000,
                        help="Maximum training steps")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name for logging")

    # Add model, SDE, backbone, and data arguments
    parser = NASEScoreModel.add_argparse_args(parser)
    parser = SDERegistry.get_by_name("ouve").add_argparse_args(parser)
    parser = BackboneRegistry.get_by_name("ncsnpp_nase").add_argparse_args(parser)
    parser = SpecsDataModule.add_argparse_args(parser)

    args = parser.parse_args()

    # Set up logging
    if args.nolog:
        logger = CSVLogger("lightning_logs")
    else:
        # Generate experiment name
        if args.exp_name is None:
            exp_name = f"nase_p{args.p_uncond}"
            if args.use_nc_loss:
                exp_name += f"_nc{args.nc_loss_weight}"
            exp_name += f"_{args.noise_encoder_type}"
        else:
            exp_name = args.exp_name

        logger = WandbLogger(
            project="sgmse-nase",
            name=exp_name,
            save_dir="./logs",
            log_model=False,
        )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=f"./logs/{logger.experiment.id if hasattr(logger, 'experiment') else 'csv'}",
            filename='step={step}',
            every_n_train_steps=args.save_ckpt_interval,
            save_top_k=-1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]

    # Create model
    model = NASEScoreModel(
        backbone=args.backbone,
        sde=args.sde,
        data_module_cls=SpecsDataModule,
        **vars(args)
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=args.gpus,
        strategy=DDPPlugin(find_unused_parameters=False) if args.gpus > 1 else 'auto',
        logger=logger,
        callbacks=callbacks,
        max_steps=args.max_steps,
        log_every_n_steps=50,
        precision=32,  # Use full precision for stability
    )

    # Train
    trainer.fit(model)


if __name__ == "__main__":
    main()
