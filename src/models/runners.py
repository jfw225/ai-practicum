import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import pandas as pd
import pickle
import random
import itertools
import time
from datetime import datetime, timedelta
from torchmetrics import ROC
from torchmetrics.classification import BinaryROC
from matplotlib import pyplot as plt

CHECKPOINT_PATH = "./checkpoints/joe_checkpoint_model.pt"


class Trainer():
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn,
        gpu_id: int,
        save_interval: int,
        metric_interval: int,
        train_data: DataLoader,
        validation_data: DataLoader = None,
        test_data: DataLoader = None
    ) -> None:
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.gpu_id = gpu_id
        self.save_interval = save_interval
        self.metric_interval = metric_interval
        self.validation_data = validation_data
        self.test_data = test_data

    def _run_batch(self, batch_tensor: torch.tensor, batch_labels: torch.tensor):
        self.optimizer.zero_grad()
        predicted_output = self.model(batch_tensor)
        # times 1.0 is to cast to float
        # loss = self.loss_fn(predicted_output,  batch_labels * 1.0)
        loss = self.loss_fn(predicted_output,  batch_labels)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch: int):
        self.model.train()
        print(f'\t[GPU {self.gpu_id}] Epoch {epoch}')
        i = 1
        all = len(self.train_data)
        for batch_tensor, batch_labels in self.train_data:
            # print(f'\t{i}/{len(self.train_data)}')
            i += 1
            batch_tensor = batch_tensor.to(self.gpu_id)
            # check batch labels type
            batch_labels = batch_labels.to(self.gpu_id)
            self._run_batch(batch_tensor, batch_labels.float())

    def _save_checkpoint(self, epoch: int):
        checkpoint = self.model.state_dict()
        torch.save(checkpoint, CHECKPOINT_PATH)
        print(f'\tModel Saved at Epoch {epoch}')

    def train(self, num_epochs: int, sv_roc: bool = False):
        # output_last = self.metric_interval < 1 or num_epochs % self.metric_interval != 0

        # output last if interval is less than 1 always
        # output last if num_epochs % self.metric_interval != 0

        # for epoch in range(1, num_epochs + 1):
        #     self._run_epoch(epoch)
        #     if self.save_interval > 0 and epoch % self.save_interval == 0:
        #         self._save_checkpoint(epoch)
        #     elif epoch == num_epochs:
        #         self._save_checkpoint(epoch)

        #     if self.metric_interval > 0 and epoch % self.metric_interval == 0:
        #         self.evaluate(self.train_data, sv_roc = True)
        #         self.evaluate(self.validation_data)

        # if output_last:
        #     self.evaluate(self.train_data, sv_roc = True)
        #     if self.validation_data != None:
        #         self.evaluate(self.validation_data)

        for epoch in range(1, num_epochs + 1):
            self._run_epoch(epoch)

            if self.save_interval > 0 and epoch % self.save_interval == 0:
                self._save_checkpoint(epoch)
            elif epoch == num_epochs:  # save last model
                self._save_checkpoint(epoch)

            if self.metric_interval > 0 and epoch % self.metric_interval == 0:
                self.evaluate(self.train_data, sv_roc=sv_roc)
                if self.validation_data != None:
                    self.evaluate(self.validation_data)
            elif epoch == num_epochs:  # Evaluate final model
                self.evaluate(self.train_data, sv_roc=sv_roc)
                if self.validation_data != None:
                    self.evaluate(self.validation_data)

    def evaluate(self, dataloader: DataLoader, sv_roc=False):
        softmax = nn.Softmax(dim=1)

        with torch.no_grad():
            self.model.eval()
            cumulative_loss = 0
            num_correct = 0
            total = 0
            num_batches = len(dataloader)
            all_preds = []  # torch.tensor([]).to(self.gpu_id)
            all_labels = []  # torch.tensor([]).to(self.gpu_id)

            for batch_tensor, batch_labels in dataloader:
                batch_tensor = batch_tensor.to(self.gpu_id)
                # check batch labels type
                batch_labels = batch_labels.to(self.gpu_id)
                predicted_output = self.model(batch_tensor)

                # times 1.0 is to cast to float
                cumulative_loss += self.loss_fn(predicted_output,
                                                batch_labels * 1.0)

                if sv_roc:
                    all_preds = torch.cat(
                        (all_preds, (softmax(predicted_output)[:, 1])))
                    all_labels = torch.cat((all_labels, batch_labels))

                else:
                    # add the predicted output to the list of all predictions
                    all_preds += predicted_output.cpu().tolist()
                    all_labels += batch_labels.cpu().tolist()

                # assuming decision boundary to be 0.5
                total += batch_labels.size(0)
                # num_correct += (torch.argmax(predicted_output, dim=1) == batch_labels).sum().item()
                num_correct += (torch.round(predicted_output)
                                == batch_labels).sum().item()

            loss = cumulative_loss/num_batches
            accuracy = num_correct/total

            d = {'predicted_output': all_preds,
                 'expected_labels': all_labels}
            df = pd.DataFrame.from_dict(d, orient='index')
            print(f'\t\t{df.to_string(header=False)}'.replace(
                'expected_labels', '\t\texpected_labels'))

            print(f'\t\tLoss: {loss} = {cumulative_loss}/{num_batches}')
            print(f'\t\tAccuracy: {accuracy} = {num_correct}/{total}')
            if sv_roc:
                Trainer.save_roc(all_preds, all_labels)
            print()

        self.model.train()

    @staticmethod
    def save_roc(all_preds, all_labels):
        # print(all_preds)
        # print(all_labels)
        roc = ROC(task="binary", thresholds=20)
        roc = BinaryROC(thresholds=20)
        all_preds = all_preds.cpu()
        all_labels = all_labels.cpu().int()
        # print("##############")
        # print(all_preds)
        # print(all_labels)
        # print("##############")
        fpr, tpr, thresholds = roc(all_preds, all_labels)
        plt.plot([0, 1], [0, 1], linestyle='dashed')
        plt.plot(fpr, tpr)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('ROC.png')


class Tester:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn = None,
        gpu_id: int = 0,
    ) -> None:
        self.loss_fn = loss_fn or torch.nn.CrossEntropyLoss()
        self.gpu_id = gpu_id
        self.model = model.to(self.gpu_id)

    def evaluate(self, dataloader: DataLoader, sv_roc=False):
        with torch.no_grad():
            self.model.eval()
            cumulative_loss = 0
            num_correct = 0
            total = 0
            num_batches = len(dataloader)
            all_preds = torch.tensor([]).to(self.gpu_id)
            all_labels = torch.tensor([]).to(self.gpu_id)

            for batch_tensor, batch_labels in dataloader:
                batch_tensor = batch_tensor.to(self.gpu_id)
                # check batch labels type
                batch_labels = batch_labels.to(self.gpu_id)
                predicted_output = self.model(batch_tensor)

                # times 1.0 is to cast to float
                cumulative_loss += self.loss_fn(predicted_output,
                                                batch_labels * 1.0)
                if sv_roc:
                    softmax = nn.Softmax(dim=1)
                    all_preds = torch.cat(
                        (all_preds, (softmax(predicted_output)[:, 1])))
                    all_labels = torch.cat((all_labels, batch_labels))

                # assuming decision boundary to be 0.5
                total += batch_labels.size(0)
                num_correct += (torch.argmax(predicted_output)
                                == batch_labels).sum().item()

            loss = cumulative_loss/num_batches
            accuracy = num_correct/total
            print(f'\t\tLoss: {loss} = {cumulative_loss}/{num_batches}')
            print(f'\t\tAccuracy: {accuracy} = {num_correct}/{total}\n\n')
            if sv_roc:
                Trainer.save_roc(all_preds, all_labels)


def testFn():
    return 5
