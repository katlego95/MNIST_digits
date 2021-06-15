import random
import torch
from torch.nn import nn
import math
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from Perceptron import Perceptron


class NN(object):
	"""docstring for NN"""
	def __init__(self, arg, size, outputs):
		super(NN, self).__init__()
		self.fc1 = nn.Linear(size, 50)
		self.fc2 = nn.Linear(50, outputs)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x
		

if __name__ == '__main__':



