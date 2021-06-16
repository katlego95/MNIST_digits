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
import errno
import os
import os.path
from os import path
import math
import PIL.Image as Image

class NN(nn.Module):

    def __init__(self,size,hidden_layers, outputs):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(size, hidden_layers)
        self.hidden = nn.Linear(in_features=hidden_layers, out_features=hidden_layers)
        self.fc2 = nn.Linear(hidden_layers, outputs)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.hidden(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':

	#GPU accelerators 
	device = torch.device('cude' if torch.cuda.is_available() else 'cpu')

	num_train = 1500
	num_valid = 1500
	input_size = 2
	epoch = 10
	learning_rate = 0.001
	input_size= 2
	hidden_layers= 10
	outputs=1
	XOR_x1 = numpy.array([0., 0., 1., 1.])
	XOR_x2 = numpy.array([0., 1., 0., 1.])
	XOR_labels  = numpy.array([0., 1., 1., 0.])

	training_examples_x1 = numpy.random.random(size = num_train)
	training_examples_x2 = numpy.random.random(size = num_train)
	validate_examples_x1 = numpy.random.random(size = num_valid)
	validate_examples_x2 = numpy.random.random(size = num_valid)
	t_labels = []
	v_labels = []
	tlabel = None
	vlabel = None

	for i in range(num_train):
		if (training_examples_x1[i] > 0.75 and training_examples_x2[i] > 0.75) or (training_examples_x1[i] < 0.75 and training_examples_x2[i] < 0.75):
			tlabel = 0.0 
		else:
			tlabel = 1.0 
		t_labels.append(tlabel)


		if (validate_examples_x1[i] > 0.75 and validate_examples_x2[i] > 0.75) or (validate_examples_x1[i] < 0.75 and validate_examples_x2[i] < 0.75):
			vlabel = 0.0 
		else:
			vlabel = 1.0 
		v_labels.append(vlabel)

	training_labels= numpy.array(t_labels)
	validate_labels = numpy.array(v_labels)

	XOR_x1 = torch.from_numpy(XOR_x1).clone().view(-1, 1)
	XOR_x2 = torch.from_numpy(XOR_x2).clone().view(-1, 1)
	XOR_labels  = torch.from_numpy(XOR_labels).clone().view(-1, 1)


	x1 = torch.from_numpy(training_examples_x1).clone().view(-1, 1)
	x2 = torch.from_numpy(training_examples_x2).clone().view(-1, 1)
	y = torch.from_numpy(training_labels).clone().view(-1, 1)

	validate_data_x1 = torch.from_numpy(validate_examples_x1).clone().view(-1, 1)
	validate_data_x2 = torch.from_numpy(validate_examples_x2).clone().view(-1, 1)
	validate_labels_y = torch.from_numpy(validate_labels).clone().view(-1, 1)
	# Combine X1 and X2
	XOR_gate = 
	training_inputs = torch.hstack([x1, x2])
	validate_inputs = torch.hstack([validate_data_x1, validate_data_x2])

	model = NN(input_size,hidden_layers,outputs)

	criterion = nn.MSELoss()
	optimizer = optim.SGD(model.parameters(), lr=learning_rate)
	#training


	for epoch in range (epoch): # 1 epoch means network has seen entire datatset
			for batch_step, (data, targets) in enumerate(training_inputs):
				data = data.to(device)
				targets = targets.to(device=device)














	#output = model(validate_inputs.float())

	#for i in range(num_train):
	#	if (i%10==0):
	#		print ("epoch: " + str(i))
	#		print ("prediction: " + str(output[i]))
	#		print ("correct answer: "+ str(validate_labels_y[i]))





































