import random
import torch
import torch.nn as nn
import math
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from Perceptron import Perceptron
import os.path


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

	Size = 28*28
	digits = 10
	learning_rate = 0.001
	batch_size = 40
	hidden_layers = 100

	path = "./data"
 
# Expand an initial ~ component
# in the given path
# using os.path.expanduser() method
	full_path = os.path.expanduser(path)


	#GPU accelerators 
	device = torch.device('cude' if torch.cuda.is_available() else 'cpu')

	#load data fron Mnist training sets 

	training_data = datasets.MNIST(root=full_path, train=True, transform=transforms.ToTensor(), download=False)
	test_data = datasets.MNIST(root=full_path, train=False,transform=transforms.ToTensor(), download=False)

	training_loader = torch.utils.data.DataLoader(dataset=training_data, batch_size=batch_size,shuffle=True, num_workers=2)
	test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size,shuffle=False, num_workers=2)







