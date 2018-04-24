# import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.autograd import Variable

data = pd.read_csv('Dataset/Absenteeism_at_work.csv')
print(data)