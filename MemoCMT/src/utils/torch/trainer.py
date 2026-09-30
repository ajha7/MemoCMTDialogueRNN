import datetime
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import mlflow
import numpy as np
import torch
import tqdm
from torch import nn
from torchsummary import summary

from ..metrics import aggregate_eval_outputs
from . import optimizers
from .callbacks import Callback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class TorchTrainer(ABC, nn.Module):
    def __init__(self, log_dir: str = "logs"):
        super().__init__()
        self.log_dir = log_dir
        self.global_step = 0
        self.start_epoch = 1

    def predict(
        self, inputs: Union[torch.Tensor, Dict, List]
    ) -> Union[torch.Tensor, Dict, List]:
        """

        Args:
            inputs (Union[torch.Tensor, Dict, List]): Inputs to the model

        Returns:
            Union[torch.Tensor, Dict, List]: Predictions
        """
        # Set model to eval mode
        self.eval()
        # Forward pass
        return self.forward(inputs)

    def train_epoch(
        self,
        step: int,
        epoch: int,
        train_data: Iterable,
        eval_data: Iterable = None,
        logger: logging.Logger = None,
        callbacks: List[Callback] = None,
    ):
        """Performs one epoch of training and validation.

        Args:
            epoch (int): Current epoch.
            train_data (Iterable): training data.
            eval_data (Iterable, optional): validation data. Defaults to None.
            logger (logging.Logger, optional): logger used for logging. Defaults to None.
            callbacks (List[Callback], optional): List of callbacks. Defaults to None.
        """
        self.network.train()
        if logger is None:
            logger = logging

        epoch_log = {}
        with tqdm.tqdm(total=len(train_data), ascii=True) as pbar:
            pbar.update(1)
            for batch in train_data:
                # Training step
                step += 1
                train_log = self.train_step(batch)
                assert isinstance(train_log, dict), "train_step should return a dict."
                # Add logs, update progress bar
                postfix = ""
                for key, value in train_log.items():
                    e_val = epoch_log.get(key, [])
                    e_val.append(value)
                    epoch_log.update({key: e_val})
                    postfix += f"{key}: {value:.4f} "

                log_every = getattr(getattr(self, "cfg", None), "log_every_n_steps", None)
                should_log = (not log_every) or (log_every <= 1) or (step % log_every == 0)
                try:
                    if should_log:
                        mlflow.log_metric(
                            "learning_rate",
                            self.optimizer.param_groups[0]["lr"],
                            step=step,
                        )
                except:
                    pass
                pbar.set_description(postfix)
                pbar.update(1)

                # Try to log learning rate if optax is implement with hyperparams injection
                try:
                    mlflow.log_metric(
                        f"learning_rate",
                        self.optimizer.param_groups[0]["lr"],
                        step=step,
                    )
                except:
                    pass

                # Callbacks
                if callbacks is not None:
                    for callback in callbacks:
                        callback(
                            self,
                            step,
                            epoch,
                            train_log,
                            isValPhase=False,
                            logger=logger,
                        )
        for key, value in epoch_log.items():
            logger.info(f"Epoch {epoch} - {key}: {np.mean(value):.4f}")
            mlflow.log_metric(f"train_epoch_{key}", np.mean(value), step=step)

        if eval_data is not None:
            logger.info("Performing validation...")
            full = self.run_eval(eval_data)
            eval_logs = {k: v for k, v in full.items() if isinstance(v, float)}
            logger.info("Validation: " + " ".join(f"{k}: {v:.4f}" for k, v in eval_logs.items()))
            for k, v in eval_logs.items():
                mlflow.log_metric(f"val_{k}", v, step=step)
            if callbacks is not None:
                for callback in callbacks:
                    callback(self, step, epoch, eval_logs, isValPhase=True, logger=logger)
        return step

    def run_eval(self, data: Iterable) -> Dict:
        """Utterance-level metrics pooled over the whole split."""
        self.network.eval()
        outputs = [self.test_step(batch) for batch in tqdm.tqdm(data, ascii=True)]
        return aggregate_eval_outputs(outputs, self.cfg.num_classes)

    def evaluate(
        self,
        test_data: Iterable,
        logger: logging.Logger = None,
    ) -> Dict:
        """Performs evaluation on the test set.

        Args:
            test_data (Iterable): Test data.
            logger (logging.Logger, optional): Logger. Defaults to None.

        Returns:
            Dict: The evaluation metrics in a dictionary with the metric name as key and the metric value as value.
        """
        self.network.eval()
        if logger is None:
            logger = logging
        test_logs = {}

        for batch in tqdm.tqdm(test_data, ascii=True):
            # Perform validation
            test_log = self.test_step(batch)
            for key, value in test_log.items():
                v = test_logs.get(key, [])
                v.append(value)
                test_logs.update({key: v})
        # Log validation metrics
        postfix = ""
        for key, value in test_logs.items():
            postfix += f"{key}: {np.mean(value):.4f} "
            try:
                mlflow.log_metric(f"test_{key}", np.mean(value), step=self.global_step)
            except:
                logger.warning(f"Could not log test metric {key} using mlflow.")
        logger.info("Test: " + postfix)

    def summary(self, input_shapes: Union[Tuple, List, Dict]):
        """Print a summary of the model.

        Args:
            input_shapes (Union[Tuple, List, Dict]): The input shapes.
        """
        summary(self, input_shapes)

    def save(self, path: str, step=None) -> str:
        """Save entire model to a checkpoint directory.

        Args:
            path (str): Path to the checkpoint directory.
            step (int, optional): Current step. Defaults to None.
        Returns:
            str: Path to the checkpoint file.
        """
        ckpt_path = os.path.join(path, "checkpoint_{}.pt".format(step))
        torch.save(self.network, ckpt_path)
        return ckpt_path

    @classmethod
    def load(self, path: str):
        """Load model from a checkpoint file.

        Args:
            path (str): Path to the checkpoint file.

        Returns:
            TorchTrainer: The loaded model.
        """
        return torch.load(path)

    def save_weights(self, path: str, step=None):
        """Save the model weights to a checkpoint directory.

        Args:
            path (str): Path to the checkpoint directory.
            step (int, optional): Current step. Defaults to None.
        """
        ckpt_path = os.path.join(path, "checkpoint_{}.pth".format(step))
        torch.save(self.network.state_dict(), ckpt_path)
        return ckpt_path

    def load_weights(self, path: str, device: str = "cpu"):
        """Load the model weights from a checkpoint file.

        Args:
            path (str): Path to the checkpoint file.
            device (str, optional): Device to load the weights on. Defaults to "cpu".
        """
        self.network.load_state_dict(torch.load(path), map_location=device)

    def save_all_states(self, path: str, global_epoch: int, global_step: int):
        checkpoint = {
            "epoch": global_epoch,
            "global_step": global_step,
            "state_dict_network": self.network.state_dict(),
            "state_optimizer": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            checkpoint["state_lr_scheduler"] = self.scheduler.state_dict()

        ckpt_path = os.path.join(
            path, "checkpoint_{}_{}.pt".format(global_epoch, global_step)
        )
        torch.save(checkpoint, ckpt_path)
        return ckpt_path

    def load_all_states(self, path: str, device: str = "cpu"):
        dict_checkpoint = torch.load(os.path.join(path), map_location=device)

        self.start_epoch = dict_checkpoint["epoch"]
        self.global_step = dict_checkpoint["global_step"]
        self.network.load_state_dict(dict_checkpoint["state_dict_network"])
        self.optimizer.load_state_dict(dict_checkpoint["state_optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])

        logging.info("Successfully loaded checkpoint from {}".format(path))
        logging.info("Resume training from epoch {}".format(self.start_epoch))

    def set_start_epoch(self, start_epoch: int):
        self.start_epoch = start_epoch

    def set_global_step(self, global_step: int):
        self.global_step = global_step

    def compile(
        self,
        optimizer: Union[str, torch.optim.Optimizer] = "sgd",
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    ):
        """Compile the model with the given optimizer.

        Args:
            optimizer (Union[str, torch.optim.Optimizer], optional): The optimizer to use. Defaults to "sgd".
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to use. Defaults to None.

        Raises:
            AttributeError: This method must be called after the model is built.
            NotImplementedError: The given optimizer is not implemented.
        """
        assert isinstance(
            optimizer, (str, torch.optim.Optimizer)
        ), "Optimizer must be a string or a torch optimizer."
        try:
            self.network
        except AttributeError:
            raise AttributeError(
                "Please add your model to the self.network attribute in the constructor!"
            )

        if type(optimizer) == str:
            available_optimizers = {
                "sgd": optimizers.sgd(
                    self.network.parameters(), learning_rate=0.01, momentum=0.9
                ),
                "adam": optimizers.adam(self.network.parameters(), learning_rate=0.01),
                "rmsprop": optimizers.rmsprop(
                    self.network.parameters(), learning_rate=0.01
                ),
                "adagrad": optimizers.adagrad(
                    self.network.parameters(), learning_rate=0.01
                ),
                "adamw": optimizers.adamw(
                    self.network.parameters(), learning_rate=0.01, weight_decay=0.01
                ),
            }
            optimizer = available_optimizers.get(optimizer, None)
            if optimizer is None:
                raise NotImplementedError(
                    "{} is not found. List of available optimizers: {}".format(
                        optimizer, list(available_optimizers.keys())
                    )
                )
        self.optimizer = optimizer
        self.scheduler = scheduler

    def fit(
        self,
        train_data: Iterable,
        epochs: int,
        eval_data: Iterable = None,
        test_data: Iterable = None,
        callbacks: List[Callback] = None,
    ):
        """Hyper API for training the model.

        Args:
            train_data (Iterable): Training data.
            epochs (int): Number of epochs to train.
            eval_data (Iterable, optional): Evaluation data. Defaults to None.
            test_data (Iterable, optional): Test data. Defaults to None.
            callbacks (List[Callback], optional): List of callbacks which will be called during training. Defaults to None.
        Raises:
            AttributeError: This method must be called after the model is compiled.
        """
        try:
            self.optimizer
        except AttributeError:
            raise AttributeError("Please compile the model first!")

        assert (
            isinstance(callbacks, list) or callbacks is None
        ), "Callbacks must be a list of Callback objects"

        # Init mlflow
        self.log_dir = os.path.join(
            self.log_dir, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        os.makedirs(self.log_dir, exist_ok=True)
        # Logger
        logging.getLogger().setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(self.log_dir, "train.log"))
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger = logging.getLogger("Training")
        logger.addHandler(file_handler)
        logger.addHandler(logging.StreamHandler())

        mlflow.set_tracking_uri(
            uri=f'file://{os.path.abspath(os.path.join(self.log_dir, "mlruns"))}'
        )
        global_step = self.global_step
        
        # Start training
        with mlflow.start_run():
            fast_cfg = getattr(self, "cfg", None)
            fast_first = bool(getattr(fast_cfg, "fast_first_epoch", False))
            skip_first_eval = bool(getattr(fast_cfg, "skip_first_epoch_eval", True))
            if not hasattr(self, "_fast_applied"):
                self._fast_applied = False
            if not hasattr(self, "_fast_backup"):
                self._fast_backup = None

            for epoch in range(self.start_epoch, epochs + 1):
                logger.info(f"Epoch {epoch}/{epochs}")
                # Apply fast-first-epoch settings
                if fast_first and epoch == self.start_epoch and not self._fast_applied:
                    self._fast_backup = {
                        "use_gradient_checkpointing": getattr(fast_cfg, "use_gradient_checkpointing", False),
                        "text_unfreeze": getattr(fast_cfg, "text_unfreeze", False),
                        "freeze_feature_extractor": getattr(fast_cfg, "freeze_feature_extractor", True),
                        "memo_chunk_size": getattr(fast_cfg, "memo_chunk_size", 16),
                    }
                    setattr(fast_cfg, "use_gradient_checkpointing", False)
                    setattr(fast_cfg, "text_unfreeze", False)
                    setattr(fast_cfg, "freeze_feature_extractor", True)
                    setattr(fast_cfg, "memo_chunk_size", max(64, self._fast_backup["memo_chunk_size"]))

                    # Freeze encoders on the live network (handles MemoCMT and MemoCMTDialogueRNN)
                    try:
                        net = self.network
                        base = getattr(net, "memo", net)
                        if hasattr(base, "text_encoder"):
                            for p in base.text_encoder.parameters():
                                p.requires_grad = False
                            base.text_encoder.eval()
                        if hasattr(base, "audio_encoder"):
                            for p in base.audio_encoder.parameters():
                                p.requires_grad = False
                            base.audio_encoder.eval()
                    except Exception:
                        pass

                    self._fast_applied = True

                # Restore after first epoch (at the start of the next epoch)
                if fast_first and epoch == self.start_epoch + 1 and self._fast_applied and self._fast_backup is not None:
                    for k, v in self._fast_backup.items():
                        setattr(fast_cfg, k, v)
                    try:
                        net = self.network
                        base = getattr(net, "memo", net)
                        # Restore requires_grad according to cfg flags
                        if hasattr(base, "text_encoder"):
                            tu = bool(getattr(fast_cfg, "text_unfreeze", False))
                            for p in base.text_encoder.parameters():
                                p.requires_grad = tu
                            base.text_encoder.train(tu)
                        if hasattr(base, "audio_encoder"):
                            au = bool(getattr(fast_cfg, "audio_unfreeze", False))
                            for p in base.audio_encoder.parameters():
                                p.requires_grad = au
                            base.audio_encoder.train(au)
                    except Exception:
                        pass
                    self._fast_applied = False
                effective_eval = None if (skip_first_eval and epoch == self.start_epoch) else eval_data
                global_step = self.train_epoch(
                    global_step, epoch, train_data, effective_eval, logger, callbacks=callbacks,
                )
                self.lr_scheduler(global_step, epoch)
                if test_data is not None:
                    self.evaluate(test_data)

    def lr_scheduler(self, step: int, epoch: int) -> None:
        """Learning rate scheduler.

        Args:
            step (int): Current step.
            epoch (int): Current epoch.
        """
        if self.scheduler is not None:
            self.scheduler.step()

    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Train step for the model.

        Args:
            batch (Dict[str, torch.Tensor]): Your inputs should be compressed into a dictionary or list.
            For example, {"inputs_1": inputs_1, "inputs_2": inputs_2} or [inputs_1, inputs_2]

        Returns:
            Dict[str, torch.Tensor]: The outputs must be a dictionary which contains the information that you want to log.
            For example, {"loss": loss, "metrics": metrics}, loss and metrics need to be a scalar.
        """
        pass

    @abstractmethod
    def test_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Test step for the model.

        Args:
            batch (Dict[str, torch.Tensor]): Your inputs should be compressed into a dictionary or list.
            For example, {"inputs_1": inputs_1, "inputs_2": inputs_2} or [inputs_1, inputs_2]

        Returns:
            Dict[str, torch.Tensor]: The outputs must be a dictionary which contains the information that you want to log.
            For example, {"loss": loss, "metrics": metrics}, loss and metrics need to be a scalar.
        """
        pass
