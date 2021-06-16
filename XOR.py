import random
from Perceptron import Perceptron


if __name__ == '__main__':


	generate_training_set = True
	num_train = 100
	generate_validation_set = True
	num_valid = 100

	training_examples = [[1.0, 1.0],
						[1.0, 0.0],
						[0.0, 1.0],
						[0.0, 0.0]]

	training_labels = [0.0, 1.0, 1.0, 1.0]

	validate_examples = training_examples
	validate_labels = training_labels

	if generate_training_set:
		training_examples = []
		training_labels = []

		for i in range(num_train):
			training_examples.append([random.random(), random.random()])
			# We want our perceptron to be noise tolerant, so we label all examples where x1 and x2 > 0.8 as 1.0
			training_labels.append(0.0 if training_examples[i][0] > 0.75 and training_examples[i][1] > 0.75 else 1.0)

	if generate_validation_set:
		validate_examples = []
		validate_labels = []

		for i in range(num_train):
			validate_examples.append([random.random(), random.random()])
			validate_labels.append(0.0 if validate_examples[i][0] > 0.75 and validate_examples[i][1] > 0.75 else 0.0)


	# Create Perceptron
	NAND1 = Perceptron(2, bias=-0.75)
	NAND2 = Perceptron(2, bias=-0.75)
	NAND3 = Perceptron(2, bias=-0.75)
	NAND4 = Perceptron(2, bias=-0.75)

	done = False

	print(NAND1.weights)
	valid_percentage1 = NAND1.validate(validate_examples, validate_labels, verbose=True)
	print(valid_percentage1)

	print(NAND2.weights)
	valid_percentage2 = NAND2.validate(validate_examples, validate_labels, verbose=True)
	print(valid_percentage1)

	print(NAND3.weights)
	valid_percentage3 = NAND3.validate(validate_examples, validate_labels, verbose=True)
	print(valid_percentage1)

	print(NAND4.weights)
	valid_percentage4 = NAND4.validate(validate_examples, validate_labels, verbose=True)
	print(valid_percentage1)

	i1 = 0
	i2 = 0
	i3 = 0
	i4 = 0
	while not done: # We want our Perceptron to have an accuracy of at least 80%
		print('Training NAND GATE_1...')
		while valid_percentage1 < 0.98:
			i1 += 1
			NAND1.train(training_examples, training_labels, 0.2)  # Train our Perceptron
			print(NAND1.weights)
			valid_percentage = NAND1.validate(validate_examples, validate_labels, verbose=True) # Validate it
			print(valid_percentage)
			if i1 == 1000: 
				break
		print('Training NAND GATE_2...')
		while valid_percentage2 < 0.98:
			i2 += 1
			NAND2.train(training_examples, training_labels, 0.2)  # Train our Perceptron
			print(NAND2.weights)
			valid_percentage = NAND2.validate(validate_examples, validate_labels, verbose=True) # Validate it
			print(valid_percentage)
			if i2 == 1000: 
				break
		print('Training NAND GATE_3...')
		while valid_percentage3 < 0.98:
			i3 += 1
			NAND3.train(training_examples, training_labels, 0.2)  # Train our Perceptron
			print(NAND3.weights)
			valid_percentage = NAND3.validate(validate_examples, validate_labels, verbose=True) # Validate it
			print(valid_percentage)
			if i3 == 1000: 
				break
		print('Training NAND GATE_4...')
		while valid_percentage4 < 0.98:
			i4 += 1
			NAND4.train(training_examples, training_labels, 0.2)  # Train our Perceptron
			print(NAND4.weights)
			valid_percentage = NAND4.validate(validate_examples, validate_labels, verbose=True) # Validate it
			print(valid_percentage)
			if i4 == 1000: 
				break
		print('Constructing Network...')
		print('Done!')
		done = True
		# This is just to break the training if it takes over 50 iterations. (For demonstration purposes)
		# You shouldn't need to do this as your networks may require much longer to train. 
		
	print('Please enter two inputs: ')








