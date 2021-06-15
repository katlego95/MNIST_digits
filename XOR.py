import random
import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets 
import math
from Perceptron import Perceptron

class NN(nn.Module):

    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(2, 3, True)
        self.fc2 = nn.Linear(3, 1, True)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':

	num_train = 100
	num_valid = 100
	input_size = 2
	epoch = 10000
	learning_rate = 0.001
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

	x1 = numpy.append (training_examples_x1, XOR_x1)
	x2 = numpy.append (training_examples_x2, XOR_x2)
	y  = numpy.append (training_labels, XOR_labels)

	x1 = torch.from_numpy(training_examples_x1).clone().view(-1, 1)
	x2 = torch.from_numpy(training_examples_x2).clone().view(-1, 1)
	y = torch.from_numpy(training_labels).clone().view(-1, 1)

	validate_data_x1 = torch.from_numpy(validate_examples_x1).clone().view(-1, 1)
	validate_data_x2 = torch.from_numpy(validate_examples_x2).clone().view(-1, 1)
	validate_labels_y = torch.from_numpy(validate_labels).clone().view(-1, 1)
	# Combine X1 and X2
	training_inputs = torch.hstack([x1, x2])
	validate_inputs = torch.hstack([validate_data_x1, validate_data_x2])

	model = NN()

	criterion = nn.MSELoss()
	optimizer = optim.SGD(model.parameters(), lr=learning_rate)
	#training
	net_loss = []
	for i in range(epoch):
		for x, target in zip(training_inputs, y):
			optimizer.zero_grad()   # zero the gradient buffers
			output = model(x.float())
			loss = criterion(output, target.float())
			loss.backward()
			optimizer.step()    # Does the update
		if (i%1000==0):
			print(i)

	output = model(validate_inputs.float())

	for i in range(num_train):
		if (i%10==0):
			print ("epoch: " + str(i))
			print ("prediction: " + str(output[i]))
			print ("correct answer: "+ str(validate_labels_y[i]))





































