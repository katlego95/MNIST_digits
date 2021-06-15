import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from Perceptron import Perceptron
import os.path
import math


class Neuralnetwork(nn.Module):
	"""docstring for Neuralnetwork"""
	def __init__(self, size,hidden_layers, outputs):
		super(Neuralnetwork, self).__init__()
		self.fc1 = nn.Linear(size, hidden_layers)
		self.hidden = nn.Linear(in_features=hidden_layers, out_features=hidden_layers)
		self.fc2 = nn.Linear(hidden_layers, outputs)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.hidden(x))
		x = self.fc2(x)
		return x

	def validate(loader, model):
		correct_samples=0
		total_samples

		with torch.no_grad():
			for x,y in loader:
				x = x.to(device)
				y = y.to(device)
		

if __name__ == '__main__':

	imageSize = 28*28
	digits = 10
	learning_rate = 0.001
	batch_size = 50
	hidden_layers = 100
	epochs=10

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

	model = Neuralnetwork(size = imageSize,hidden_layers=hidden_layers, outputs = digits)

	#loss and optimizer 
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	#training gym

	data_size = len(training_loader)

	for epoch in range (epochs): # 1 epoch means network has seen entire datatset
		for batch_step, (data, targets) in enumerate(training_loader):
			data = data.reshape(-1, imageSize).to(device)
			targets = targets.to(device=device)

			output= model(data)
			loss = criterion(output, targets)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			print('Epoch: {}/{}, step: {}/{}, loss: {:.4f}'.format(epoch+1, epochs,batch_step, data_size, loss.item()))










