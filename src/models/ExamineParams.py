import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os 
import pandas as pd
import pickle
import random
import itertools
import time
from datetime import datetime, timedelta




def main():
    state_dict = dict(torch.load('checkpoint_model.pt'))



    print(list(state_dict.keys()))
    



if __name__ == "__main__":
    main()