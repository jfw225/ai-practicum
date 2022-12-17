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

# CHECKPOINT_PATH = "./checkpoints/joe_checkpoint_model.pt"
# CHECKPOINT_PATH = "./checkpoints/model_1.pt"
# CHECKPOINT_PATH = "./checkpoints/model_2.pt"
# CHECKPOINT_PATH = "./checkpoints/model_3.pt"
# CHECKPOINT_PATH = "./checkpoints/model_4.pt"

# CHECKPOINT_PATH = "./checkpoints/model_5.pt"
# CHECKPOINT_PATH = "./checkpoints/model_6.pt"
# CHECKPOINT_PATH = "./checkpoints/model_7.pt"
# CHECKPOINT_PATH = "./checkpoints/model_8.pt"

# CHECKPOINT_PATH = "./checkpoints/model_9.pt"
# CHECKPOINT_PATH = "./checkpoints/model_10.pt"
# CHECKPOINT_PATH = "./checkpoints/model_11.pt"
# CHECKPOINT_PATH = "./checkpoints/model_12.pt"


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
        # print(predicted_output)
        # print(batch_labels)
        loss = self.loss_fn(predicted_output, batch_labels.long())
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch: int):
        self.model.train()
        print(f'\t[GPU {self.gpu_id}] Epoch {epoch}')
        # i = 1
        # all = len(self.train_data)
        for batch_tensor, batch_labels in self.train_data:
            # print(f'\t{i}/{len(self.train_data)}')
            # i += 1
            batch_tensor = batch_tensor.to(self.gpu_id)
            # check batch labels type
            batch_labels = batch_labels.to(self.gpu_id)
            self._run_batch(batch_tensor, batch_labels.float())

    def _save_checkpoint(self, epoch: int):
        checkpoint = self.model.state_dict()
        torch.save(checkpoint, CHECKPOINT_PATH)
        print(f'\tModel Saved at Epoch {epoch}')

    def train(self, num_epochs: int, sv_roc: bool = False):

        for epoch in range(1, num_epochs + 1):
            self._run_epoch(epoch)

            if self.save_interval > 0 and epoch % self.save_interval == 0:
                self._save_checkpoint(epoch)
            elif epoch == num_epochs:  # save last model
                self._save_checkpoint(epoch)

            if self.metric_interval > 0 and epoch % self.metric_interval == 0:
                print("\tTrain Metrics:")
                self.evaluate(self.train_data, sv_roc=sv_roc)
                if self.validation_data != None:
                    print("\tTest Metrics:")
                    self.evaluate(self.validation_data)
            elif epoch == num_epochs:  # Evaluate final model
                print("\tTrain Metrics:")
                self.evaluate(self.train_data, sv_roc=sv_roc)
                if self.validation_data != None:
                    print("\tTest Metrics:")
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
            all_preds2 = []
            all_preds3 = []
            all_preds4 = []
            all_labels = []  # torch.tensor([]).to(self.gpu_id)

            for batch_tensor, batch_labels in dataloader:
                batch_tensor = batch_tensor.to(self.gpu_id)
                # check batch labels type
                batch_labels = batch_labels.to(self.gpu_id).long()
                predicted_output = self.model(batch_tensor)

                # times 1.0 is to cast to float
                cumulative_loss += self.loss_fn(predicted_output, batch_labels)

                if sv_roc:
                    all_preds = torch.cat(
                        (all_preds, (softmax(predicted_output)[:, 1])))
                    all_labels = torch.cat((all_labels, batch_labels))

                else:
                    # add the predicted output to the list of all predictions

                    # all_preds += predicted_output.cpu().tolist()

                    # all_preds += ((torch.softmax(predicted_output,
                    #               dim=1)[:, 0])).cpu().tolist()
                    # all_preds2 += ((torch.softmax(predicted_output,
                    #                dim=1)[:, 1])).cpu().tolist()

                    # all_preds3 += (predicted_output[:, 0]).cpu().tolist()
                    # all_preds4 += (predicted_output[:, 1]).cpu().tolist()

                    # all_labels += batch_labels.cpu().tolist()

                    # assuming decision boundary to be 0.5
                    pass

                total += batch_labels.size(0)
                num_correct += (torch.argmax(predicted_output,
                                dim=1) == batch_labels).sum().item()
                # num_correct += (torch.round(predicted_output)
                #                 == batch_labels).sum().item()

            loss = cumulative_loss/num_batches
            accuracy = num_correct/total

            # print(all_preds)
            # print(all_labels)

            # d = {'predicted_output': all_preds,
            #      'predicted_output2': all_preds2,
            #      'predicted_output3': all_preds3,
            #      'predicted_output4': all_preds4,
            #      'expected_labels': all_labels}
            # df = pd.DataFrame.from_dict(d, orient='index')
            # print(f'\t\t{df.to_string(header=False)}'.replace(
            #     'expected_labels', '\t\texpected_labels').replace('predicted_output2', '\t\tpredicted_output2').replace('predicted_output3', '\t\tpredicted_output3').replace('predicted_output4', '\t\tpredicted_output4'))

            print(f'\t\tLoss: {loss} = {cumulative_loss}/{num_batches}')
            print(f'\t\tAccuracy: {accuracy} = {num_correct}/{total}')

            if sv_roc:
                Trainer.save_roc(all_preds, all_labels)
            print()

        self.model.train()

    @ staticmethod
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
