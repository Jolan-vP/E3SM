"""Base trainer modules for pytorch models.

Classes
---------
BaseTrainer()
EarlyStopping()

"""

import torch
import sys
from abc import abstractmethod
from numpy import inf
import time
import copy
import numpy as np
from utils import MetricTracker


class BaseTrainer:
    """
    Base class for all trainers.
    """

    def __init__(
        self,
        model,
        criterion,
        metric_funcs,
        optimizer,
        max_epochs,
        config
    ):

        self.config = config

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.max_epochs = max_epochs
        self.early_stopper = EarlyStopping(**config["trainer"]["early_stopping"]["args"])

        self.metric_funcs = metric_funcs
        self.batch_log = MetricTracker(
            "batch",
            "loss",
            "val_loss",
            *[m.__name__ for m in self.metric_funcs],
            *["val_" + m.__name__ for m in self.metric_funcs],
        )
        self.log = MetricTracker(
            "epoch",
            "loss",
            "val_loss",
            *[m.__name__ for m in self.metric_funcs],
            *["val_" + m.__name__ for m in self.metric_funcs],
        )

    def fit(self, std_mean):
        """
        Full training logic
        """

        for epoch in range(self.max_epochs + 1):

            start_time = time.time()

            outputs = self._train_epoch(epoch)

            # log the results of the epoch
            self.batch_log.result()
            self.log.update("epoch", epoch)
            for key in self.batch_log.history:
                self.log.update(key, self.batch_log.history[key])

            # Every 3 epochs save best model every three epochs
            if epoch % 3 == 0:
                path = str(self.config["perlmutter_model_dir"]) + str(self.config["expname"]) + ".pth"
                torch.save({
                            "model_state_dict" : self.model.state_dict(),
                            "training_std_mean" : std_mean,
                            }, path)


            # early stopping
            if self.early_stopper.check_early_stop(epoch, self.log.history["val_loss"][epoch], self.model, outputs):
                print(
                    f"Restoring model weights from the end of the best epoch {self.early_stopper.best_epoch}: "
                    f"val_loss = {self.early_stopper.min_validation_loss:.5f}"
                )
                self.log.print(idx=self.early_stopper.best_epoch)

                self.model.load_state_dict(self.early_stopper.best_model_state)
                self.model.eval()

                break

            # Print out progress during training
            # original_stdout = sys.stdout
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(
                f"Epoch {epoch:3d}/{self.max_epochs:2d}\n"
                f"  {elapsed_time:.1f}s"
                f" - loss: {self.log.history['loss'][epoch]:.5f}"
                f" - val_loss: {self.log.history['val_loss'][epoch]:.5f}"
            )

            # with open('/Users/C830793391/Documents/Research/E3SM/saved/output/logs/' + str(self.config["expname"] + "_logs.txt"), 'w') as f:
            # # Redirect standard output to the file
            #     sys.stdout = f

            #     # Print some output
            #     print(
            #     f"Epoch {epoch:3d}/{self.max_epochs:2d}\n"
            #     f"  {elapsed_time:.1f}s"
            #     f" - loss: {self.log.history['loss'][epoch]:.5f}"
            #     f" - val_loss: {self.log.history['val_loss'][epoch]:.5f}"
            # )

        # reset the batch_log
        self.batch_log.reset()

    @abstractmethod
    def _train_epoch(self):
        """
        Train an epoch

        """
        raise NotImplementedError

    @abstractmethod
    def _validation_epoch(self):
        """
        Validate after training an epoch

        """
        raise NotImplementedError


class EarlyStopping:
    """
    Base class for early stopping.
    """

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float("inf")
        self.min_validation_loss = float("inf")
        self.best_model_state = None
        self.best_epoch = None

    def check_early_stop(self, epoch, validation_loss, model, outputs):
        if validation_loss < (self.min_validation_loss - self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0

            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.best_outputs = outputs
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False