import random
import torch
import torch.nn as nn
import math
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from Perceptron import Perceptron
import os.path


class NN(object):
	"""docstring for NN"""
	def __init__(self, size,hidden_layers, outputs):
		super(NN, self).__init__()
		self.fc1 = nn.Linear(size, hidden_layers)
		self.hidden = nn.Linear(in_features=hidden_layers, out_features=hidden_layers)
		self.fc2 = nn.Linear(hidden_layers, outputs)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = 
		x = self.fc2(x)
		return x
		

if __name__ == '__main__':

	imageeSize = 28*28
	digits = 10
	learning_rate = 0.001
	batch_size = 50
	hidden_layers = 100
	epochs=1

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

	training_loader = DataLoader(dataset=training_data, batch_size=batch_size,shuffle=True, num_workers=2)
	test_loader = DataLoader(dataset=test_data, batch_size=batch_size,shuffle=False, num_workers=2)

	#check_data = iter(training_loader)
	#img, lab = next(check_data)
	#print (img.shape, lab.shape)

	model = NN(size = imageeSize, outputs = digits)

	#loss and optimizer 
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	#training gym

	for epoch in range (epochs): # 1 epoch means network has seen entire datatset
		for batche_idx, (data, targets) in enumrate(training_loader):
			data = data.to(device=device)
			targets = targets.to(device=device)
			print(data.shape)










