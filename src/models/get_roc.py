import torch
import random
from models import ConvLSTM
from torch import nn
from runners import Tester
from data import get_train_test_dataloader
from torchmetrics.classification import BinaryROC
from matplotlib import pyplot as plt
import os
from sklearn.metrics import roc_auc_score


def load_model(model, model_path) -> Tester:
    """
    Loads the model and returns a runner object.
    """

    # load the model weights
    print(model_path)
    model.load_state_dict(torch.load(model_path))

    # create the test runner
    tester = Tester(model=model)

    return tester


def get_test_preds_labels(trainer, dataloader):
    # softmax = nn.Softmax(dim=1)

    all_preds = torch.tensor([]).to('cuda')
    all_labels = torch.tensor([]).to('cuda')

    all_preds = []
    all_labels = []

    for batch_tensor, batch_labels in dataloader:
        batch_tensor = batch_tensor.to('cuda')
        # check batch labels type
        batch_labels = batch_labels.to('cuda').long()
        predicted_output = trainer.model(batch_tensor)

        # all_preds = torch.cat((all_preds, (softmax(predicted_output)[:, 1])))
        # all_labels = torch.cat((all_labels, batch_labels))

        all_preds += ((torch.softmax(predicted_output, dim=1)
                      [:, 1])).cpu().tolist()
        all_labels += batch_labels.cpu().tolist()

    return all_preds, all_labels


def save_roc(all_preds, all_labels, path, name):

    # print(all_preds)
    # print(all_labels)
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
    plt.title(f'{name}')
    plt.savefig(f'curves/{name}_ROC.png')


def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #  '3' -> GPU 1
    #  '1' -> GPU 2
    #  '0' -> GPU 3
    #  '2' -> GPU 0

    batch_size = 2
    training_generator, test_generator = get_train_test_dataloader(
        (0.8, 0.2), batch_size, balance=True)

    models = ['model_1', 'model_2', 'model_3', 'model_4', 'model_5', 'model_6',
              'model_7', 'model_8', 'model_9', 'model_10', 'model_11', 'model_12']

    i = 1
    for m in models:
        model = ConvLSTM(conv_kernel=3, pool_kernel=2,
                         input_dim=192, output_dim=192)

        trainer = load_model(model, f'./checkpoints/{m}.pt')

        all_preds, all_labels = get_test_preds_labels(trainer, test_generator)

        # all_preds = all_preds.tolist()
        # all_labels = all_labels.tolist()

        all_labels = list(map(lambda x: int(x), all_labels))
        print(all_labels)

        curr_roc_score = roc_auc_score(all_labels, all_preds)

        print(f'\tModel 1 AUC ROC: {curr_roc_score}')
        i += 1


if __name__ == '__main__':
    main()
