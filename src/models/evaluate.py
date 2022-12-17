import torch
import random
from models import ConvLSTM
from runners import Tester
from data import get_constant_data, get_train_test_dataloader


def load_model(model, model_path) -> Tester:
    """
    Loads the model and returns a runner object.
    """

    # load the model weights
    model.load_state_dict(torch.load(model_path))

    # create the test runner
    tester = Tester(model=model)

    return tester


def test_model(tester, data_generator):
    tester.evaluate(data_generator, sv_roc=False)


if __name__ == '__main__':
    path = "./joe_checkpoint_model.pt"

    # get the data
    data = get_constant_data()

    model = ConvLSTM(conv_kernel=3, pool_kernel=2,
                     input_dim=192, output_dim=192)

    tester = load_model(model, path)

    test_model(tester, data)
