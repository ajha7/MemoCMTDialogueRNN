import logging
import os
import sys

lib_path = os.path.abspath("").replace("scripts", "src")
sys.path.append(lib_path)

import argparse
import datetime
import random

import numpy as np
import torch
from torch import optim

import trainer as Trainer
from configs.base import Config
from data.dataloader import build_train_test_dataset
from models import losses, networks, optims
from utils.configs import get_options, parse_overrides
from utils.torch.callbacks import CheckpointsCallback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(cfg: Config):
    set_seed(getattr(cfg, "seed", 0))
    logging.info("Initializing model...")
    # Model
    try:
        network = getattr(networks, cfg.model_type)(cfg)
        network.to(device)
    except AttributeError:
        raise NotImplementedError("Model {} is not implemented".format(cfg.model_type))

    logging.info("Initializing checkpoint directory and dataset...")
    # Preapre the checkpoint directory
    cfg.checkpoint_dir = checkpoint_dir = os.path.join(
        os.path.abspath(cfg.checkpoint_dir),
        cfg.name,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    log_dir = os.path.join(checkpoint_dir, "logs")
    weight_dir = os.path.join(checkpoint_dir, "weights")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)
    cfg.save(cfg)

    # Build datasets first so we can derive class weights if needed
    train_ds, test_ds = build_train_test_dataset(cfg)

    # Compute class weights from training set for imbalanced ERC (conversation-aware only)
    if getattr(cfg, "conversation_aware", False):
        try:
            import torch
            num_classes = cfg.num_classes
            counts = torch.zeros(num_classes, dtype=torch.long)
            dataset = train_ds.dataset  # remember, this is the conversation dataset
            for conv in dataset.conversations:
                for _, _, _, _, lab in conv:
                    counts[int(lab)] += 1
            weights = (counts.sum() / counts.float().clamp_min(1)) / num_classes
            cfg.class_weights = weights.tolist()
            logging.info(f"Computed class weights: {cfg.class_weights}")
        except Exception as e:
            logging.warning(f"Could not compute class weights automatically: {e}")

    try:
        criterion = getattr(losses, cfg.loss_type)(
            cfg,
            label_smoothing=getattr(cfg, "label_smoothing", 0.0),
        )
        criterion.to(device)
    except AttributeError:
        raise NotImplementedError("Loss {} is not implemented".format(cfg.loss_type))

    try:
        trainer = getattr(Trainer, cfg.trainer)(
            cfg=cfg,
            network=network,
            criterion=criterion,
            log_dir=cfg.checkpoint_dir,
        )
    except AttributeError:
        raise NotImplementedError("Trainer {} is not implemented".format(cfg.trainer))
    logging.info("Initializing trainer...")

    logging.info("Start training...")

    optimizer = optims.get_optim(cfg, network)
    lr_scheduler = None
    if cfg.learning_rate_step_size is not None:
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.learning_rate_step_size,
            gamma=cfg.learning_rate_gamma,
        )

    ckpt_callback = CheckpointsCallback(
        checkpoint_dir=weight_dir,
        save_freq=cfg.save_freq,
        max_to_keep=cfg.max_to_keep,
        save_best_val=cfg.save_best_val,
        save_all_states=cfg.save_all_states,
        monitor=[getattr(cfg, "best_metric", "ua")],
    )

    if cfg.resume:
        trainer.load_all_states(cfg.resume_path)

    trainer.compile(optimizer=optimizer, scheduler=lr_scheduler)
    trainer.fit(train_ds, cfg.num_epochs, test_ds, callbacks=[ckpt_callback])

    if getattr(cfg, "conversation_aware", False):
        import json
        from data.dataloader_dialogue import build_conversation_eval_loader

        best_metric = getattr(cfg, "best_metric", "ua")
        best_path = os.path.join(weight_dir, f"best_{best_metric}", "checkpoint_0.pth")
        trainer.network.load_state_dict(torch.load(best_path, map_location=device))
        test = trainer.run_eval(build_conversation_eval_loader(cfg, "test.pkl"))
        results = {
            "name": cfg.name,
            "seed": getattr(cfg, "seed", 0),
            "context_window": getattr(cfg, "context_window", None),
            "ablate_audio": getattr(cfg, "ablate_audio", False),
            "ablate_text": getattr(cfg, "ablate_text", False),
            "best_val": ckpt_callback.best_val,
            "test": test,
        }
        logging.info(f"TEST ({best_metric}-selected): " + json.dumps({k: v for k, v in test.items() if isinstance(v, float)}))
        for path in filter(None, [os.path.join(checkpoint_dir, "results.json"), getattr(cfg, "results_path", None)]):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w") as f:
                json.dump(results, f, indent=2)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", type=str, default="../src/configs/base.py")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    cfg: Config = get_options(args.config)
    for key, value in parse_overrides(args.set).items():
        setattr(cfg, key, value)
    if cfg.resume and cfg.cfg_path is not None:
        resume = cfg.resume
        resume_path = cfg.resume_path
        cfg.load(cfg.cfg_path)
        cfg.resume = resume
        cfg.resume_path = resume_path

    main(cfg)
