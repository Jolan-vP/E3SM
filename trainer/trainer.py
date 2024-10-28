"""Trainer modules for pytorch models.

Classes
---------
Trainer(base.base_trainer.BaseTrainer)

"""

import numpy as np
import torch
from base.base_trainer import BaseTrainer
from utils import MetricTracker
from model.build_model import TorchModel


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metric_funcs,
        optimizer,
        max_epochs,
        data_loader,
        validation_data_loader,
        device,
        config,
    ):
        super().__init__(
            model,
            criterion,
            metric_funcs,
            optimizer,
            max_epochs,
            config,
        )
        self.config = config
        self.device = device
        self.model = model.to(device)
        self.data_loader = data_loader
        self.validation_data_loader = validation_data_loader

        self.do_validation = True

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model.train()
        self.batch_log.reset()

        # counter = 0
        # transition_detected = False


        for batch_idx, (data, target) in enumerate(self.data_loader):

            input, target = (
                data[0].to(self.device),
                target.to(self.device)
            )


            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            output = self.model(input)

            # print(f"Outputs: {output}")
            # Compute the loss and its gradients

            loss = self.criterion(output, target)
            loss.backward()

            # l1_weight_grad = self.model.L1.weight.grad
            # if l1_weight_grad is not None:
            #     print(f"Iteration {counter}: Gradient for L1.weight: {l1_weight_grad}")
            #     if not transition_detected:
            #         if torch.isnan(l1_weight_grad).any():
            #             print(f"Transition to NaN detected at iteration {counter}")
            #             transition_detected = True
            # counter += 1

            # # Print gradients
            # for name, param in self.model.named_parameters():
            #     # if name == "L1.weight":
            #     if param.grad is not None:
            #         print(f"Gradient for {name}: {param.grad}")
                    
            # Clip gradients to reduce size?
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

            # Adjust learning weights
            self.optimizer.step()

            # Log the results
            self.batch_log.update("batch", batch_idx)
            self.batch_log.update("loss", loss.item())
            for met in self.metric_funcs:
                self.batch_log.update(met.__name__, met(output, target))

            


        
        # Run validation
        if self.do_validation:
            self._validation_epoch(epoch)

        return 

    def _validation_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        outputs = []
        with torch.no_grad():

            for batch_idx, (data, target) in enumerate(self.validation_data_loader):
                input, target = (
                    data[0].to(self.device),
                    target.to(self.device)
                ) 


                output = self.model(input)
                loss = self.criterion(output, target)
                
                # Log the results
                self.batch_log.update("val_loss", loss.item())
                for met in self.metric_funcs:
                    self.batch_log.update("val_" + met.__name__, met(output, target))

        return 