import random
from Perceptron import Perceptron

def classification(GATE_1,GATE_2,GATE_3,GATE_4,user_input):

	NAND1_output = GATE_1.activate(user_input)

	NAND2_input = (user_input[0],NAND1_output)

	NAND2_output = GATE_2.activate(NAND2_input)

	NAND3_input = (user_input[1],NAND1_output)

	NAND3_output = GATE_3.activate(NAND3_input)

	NAND4_input = (NAND2_output,NAND3_output)

	NAND4_output = GATE_4.activate(NAND4_input)

	print('XOR Gate: ' + str(NAND4_output))


if __name__ == '__main__':

	generate_training_set = True
	num_train = 800
	generate_validation_set = True
	num_valid = 800
	epoch = 50000

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
		half = num_train/2

		for i in range(100):
			training_examples.append([random.uniform(0.0, 0.74), random.uniform(0.0, 0.74)])

		for i in range(100):
			training_examples.append([random.uniform(0.75, 1), random.uniform(0.0, 0.74)])

		for i in range(100):
			training_examples.append([random.uniform(0.0, 0.74), random.uniform(0.75, 1.0)])

		for i in range(100):
			training_examples.append([random.uniform(0.75, 1.0), random.uniform(0.75, 1.0)])

		for i in range(100):
			training_examples.append([random.random(), random.random()])

		for i in range(100):
			training_examples.append([random.uniform(-1.0, 0.0), random.uniform(0.75, 1.0)])

		for i in range(100):
			training_examples.append([random.uniform(0.75, 1.0), random.uniform(-1.0, 0.0)])

		for i in range(100):
			training_examples.append([random.uniform(-1.0, 0.0), random.uniform(-1.0, 0.0)])

		random.shuffle(training_examples)
		
		for i in range(num_train):
			if (training_examples[i][0] >= 0.75 and training_examples[i][1] >= 0.75): 
				t_label=0.0
			else:
				t_label=1.0
			# We want our perceptron to be noise tolerant, so we label all examples where x1 and x2 > 0.8 as 1.0
			training_labels.append(t_label)

	if generate_validation_set:
		validate_examples = []
		validate_labels = []

		for i in range(100):
			validate_examples.append([random.uniform(0.0, 0.74), random.uniform(0.0, 0.74)])

		for i in range(100):
			validate_examples.append([random.uniform(0.75, 1), random.uniform(0.0, 0.74)])

		for i in range(100):
			validate_examples.append([random.uniform(0.0, 0.74), random.uniform(0.75, 1.0)])

		for i in range(100):
			validate_examples.append([random.uniform(0.75, 1), random.uniform(0.75, 1.0)])

		for i in range(100):
			validate_examples.append([random.random(), random.random()])

		for i in range(100):
			validate_examples.append([random.uniform(-1.0, 0.0), random.uniform(0.75, 1.0)])

		for i in range(100):
			validate_examples.append([random.uniform(0.75, 1.0), random.uniform(-1.0, 0.0)])

		for i in range(100):
			validate_examples.append([random.uniform(-1.0, 0.0), random.uniform(-1.0, 0.0)])

		random.shuffle(validate_examples)

		for i in range(num_valid):
			if (validate_examples[i][0] >= 0.75 and validate_examples[i][1] >= 0.75): 
				v_label=0.0
			else:
				v_label=1.0
			# We want our perceptron to be noise tolerant, so we label all examples where x1 and x2 > 0.8 as 1.0
			validate_labels.append(v_label)

	test_d = []
	validate_d = []

	for i in range(num_train):

		train_d=(training_examples[i][0],training_examples[i][1],training_labels[i])
		vali_d=(validate_examples[i][0],validate_examples[i][1],validate_labels[i])

		test_d.append(train_d)
		validate_d.append(vali_d)

	

	# Create Perceptron
	NAND1 = Perceptron(2, bias=1.0)
	NAND2 = Perceptron(2, bias=1.0)
	NAND3 = Perceptron(2, bias=1.0)
	NAND4 = Perceptron(2, bias=1.0)

	done = False
	input1= None
	learning_rate = 0.01

	#print(NAND1.weights)
	valid_percentage1 = NAND1.validate(validate_examples, validate_labels, verbose=False)

	#print(NAND2.weights)
	valid_percentage2 = NAND2.validate(validate_examples, validate_labels, verbose=False)
	

	#print(NAND3.weights)
	valid_percentage3 = NAND3.validate(validate_examples, validate_labels, verbose=False)
	

	#print(NAND4.weights)
	valid_percentage4 = NAND4.validate(validate_examples, validate_labels, verbose=False)
	

	print('NAND GATE 1 accuracy : ' +str(valid_percentage1))
	print('NAND GATE 2 accuracy : ' +str(valid_percentage2))
	print('NAND GATE 3 accuracy : ' +str(valid_percentage3))
	print('NAND GATE 4 accuracy : ' +str(valid_percentage4))

	i1 = 0
	i2 = 0
	i3 = 0
	i4 = 0
	while not done: # We want our Perceptron to have an accuracy of at least 80%
		print('Training NAND GATE_1...')
		while valid_percentage1 < 0.98:
			i1 += 1
			NAND1.train(training_examples, training_labels, learning_rate)  # Train our Perceptron
			valid_percentage1 = NAND1.validate(validate_examples, validate_labels, verbose=False) # Validate it
			if i1 == epoch: 
				break
		print(i1)
		print('------------------------------------------------------')
		print('Training NAND GATE_2...')
		while valid_percentage2 < 0.98:
			i2 += 1
			NAND2.train(training_examples, training_labels, learning_rate)  # Train our Perceptron
			valid_percentage2 = NAND2.validate(validate_examples, validate_labels, verbose=False) # Validate it
			if i2 == epoch:
				break
		print(i2)
		print('------------------------------------------------------')
		print('Training NAND GATE_3...')
		while valid_percentage3 < 0.98:
			i3 += 1
			NAND3.train(training_examples, training_labels, learning_rate)  # Train our Perceptron
			valid_percentage3 = NAND3.validate(validate_examples, validate_labels, verbose=False) # Validate it
			if i3 == epoch: 
				break
		print(i3)
		print('------------------------------------------------------')
		print('Training NAND GATE_4...')
		while valid_percentage4 < 0.98:
			i4 += 1
			NAND4.train(training_examples, training_labels, learning_rate)  # Train our Perceptron
			valid_percentage4 = NAND4.validate(validate_examples, validate_labels, verbose=False) # Validate it
			if i4 == epoch: 
				break
		print(i4)
		print('Done!')
		print('------------------------------------------------------')
		done = True
		# This is just to break the training if it takes over 50 iterations. (For demonstration purposes)
		# You shouldn't need to do this as your networks may require much longer to train. 
		

	print('NAND GATE 1 accuracy after training : ' +str(valid_percentage1))
	print('NAND GATE 2 accuracy after training : ' +str(valid_percentage2))
	print('NAND GATE 3 accuracy after training : ' +str(valid_percentage3))
	print('NAND GATE 4 accuracy after training : ' +str(valid_percentage4))
	print('Constructing Network...')
	print('------------------------------------------------------')
	print('Please enter two inputs: ')
	print('type exit to terminate programme')

	while (input1 !='exit'):

		input1 = input()

		if (input1=='exit' or input1=='Exit' or input1=='EXIT'):
			break

		x, y = map(float, input1.split())
		user_example = (x,y)

		classification(NAND1,NAND2,NAND3,NAND4, user_example)
		
		print('Please enter two inputs: ')








