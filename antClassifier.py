#--------------------------------------------------------------------------
#-------------------------- Library Imports -------------------------------
#--------------------------------------------------------------------------

import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import time
import os
import random
import matplotlib.pyplot as plt
from matplotlib import colors

#--------------------------------------------------------------------------
#-------------------------- Accessing Data --------------------------------
#--------------------------------------------------------------------------

labelledImages = 