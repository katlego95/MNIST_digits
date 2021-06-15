import random
import matplotlib.pyplot as plt
import numpy
import torch
import torchvision
import math
from torchvision import transforms, datasets 
from Perceptron import Perceptron

if __name__ == '__main__':

	num_train = 1000
	num_valid = 1000
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


	training_data_x1 = torch.from_numpy(training_examples_x1).clone().view(-1, 1)
	training_data_x2 = torch.from_numpy(training_examples_x2).clone().view(-1, 1)
	training_labels_y = torch.from_numpy(training_labels).clone().view(-1, 1)

	validate_data_x1 = torch.from_numpy(validate_examples_x1).clone().view(-1, 1)
	validate_data_x2 = torch.from_numpy(validate_examples_x2).clone().view(-1, 1)
	validate_labels_y = torch.from_numpy(validate_labels).clone().view(-1, 1)

	# Combine X1 and X2
  training_inputs = torch.hstack([training_data_x1, training_data_x2])
  validate_inputs = torch.hstack([validate_data_x1, validate_data_x2])


























