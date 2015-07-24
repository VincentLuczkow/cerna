# Builds a matrix representing tests from an overdetermined model of the network.
def build_overdetermined_knockout_gamma_model(self, mRNA, miRNA):
	actual_n = self.x + self.y
	new_nodes_total = mRNA + miRNA
	n = self.x + self.y + mRNA + miRNA
	matrix = np.ones([2 * n, 2 * n])

	# Copy actual network data over
	# Knockouts
	set_sub_matrix(range(2 * actual_n - 1), range(actual_n), matrix, self.knockout_gamma_matrix)
	set_sub_matrix(range(2 * actual_n - 1), range(n, n + actual_n), matrix,
				   get_sub_matrix(range(2 * actual_n), range(actual_n, 2 * actual_n), self.knockout_gamma_matrix))
	# Gammas
	# Add in extra nodes knockout data
	ks, mus, gammas = nf.generate_parameters(mRNA, miRNA)
	for j in range(new_nodes_total):
		extra_node_mean = ks[j] / mus[j]

		# Add in extra node data
		for i in range(2 * n - 1):
			matrix[i][n + actual_n + j] = np.random.normal(extra_node_mean, extra_node_mean * .02)

		# Knockout tests for current extra node
		matrix[2 * actual_n - 1 + j][n + actual_n + j] = 0
		matrix[2 * actual_n - 1 + j][actual_n + j] = 0

	# Copy slightly randomized wild type data over into extra tests
	for i in range(2 * new_nodes_total):
		for j in range(actual_n):
			wild_type_mean = self.knockout_gamma_matrix[0][j + actual_n]
			matrix[2 * actual_n - 1 + i][n + j] = np.random.normal(wild_type_mean, wild_type_mean * .02)

	matrix[2 * n - 1] = np.zeros(2 * n)
	return matrix


def base_network_simulation(ks, mus, gammas, x, y, knockouts, gamma_changes, lambda_changes, sim):
	n = x + y
	means = np.zeros(n)

	# NF.create_base_knockout_network_file(deepcopy(x), deepcopy(y), deepcopy(ks), deepcopy(mus), deepcopy(gammas), knockouts)
	sim.Model("RNAKnockoutNetwork" + str(x) + "," + str(y))

	for g in gamma_changes:
		sim.ChangeParameter("gammar" + str(g[0]) + "r" + str(g[1]), g[2])

	for l in lambda_changes:
		sim.ChangeParameter("k" + str(l[0]), l[1])

	sim.DoCompleteStochSim()
	data = sim.data_stochsim.species_means
	for i in range(n):
		# If we didn't knock it out
		try:
			means[i] = data["r" + str(i)]
		except IndexError:
			continue
	return means


def build_incomplete_gamma_model(self, mRNA, miRNA):
	n = self.x + self.y
	t = miRNA + mRNA
	rows = random.sample(range(n), t)
	rows.sort()
	columns = deepcopy(rows)
	columns.extend(map(lambda x: x + n, rows))

	rows_map = {}
	columns_map = {}

	for i in range(len(rows)):
		rows_map[rows[i]] = i
	for i in range(len(columns)):
		columns_map[columns[i]] = i
	return rows, columns, rows_map, columns_map


def breaking_gamma_test(self, t, old_correct_model_tests, old_incomplete_model_tests, columns_map):
	n = self.x + self.y

	# Data to return
	correct_model = {}
	incomplete_model = {}
	# New test data
	correct_model_tests = np.zeros([n + 1, 2 * n])
	incomplete_model_tests = np.zeros([t + 1, 2 * t])
	# New matrices that include new test
	correct_model_matrix = np.zeros([n + 1, n])
	incomplete_model_matrix = np.zeros([t + 1, t])

	# Run an additional test
	additional_gamma_test = np.ones([1, 2 * n])
	gamma_changes = []
	mRNA = random.choice(range(self.x))
	for miRNA in self.gammas[mRNA]:
		new_gamma_value = np.random.uniform(0.1, 10) * self.gammas[mRNA][miRNA]
		gamma_changes.append((mRNA, miRNA, new_gamma_value))
	additional_gamma_test[0][n:] = base_network_ode_solution(self.ks, self.mus, self.gammas, self.x, self.y, [],
															 gamma_changes, [], deepcopy(self.sim))

	# Copy old tests
	set_sub_matrix(range(n), range(2 * n), correct_model_tests, old_correct_model_tests)
	set_sub_matrix(range(t), range(2 * t), incomplete_model_tests, old_incomplete_model_tests)
	# Incorporate the new test
	correct_model_tests[n] = additional_gamma_test
	incomplete_model_tests[t] = get_sub_matrix([0], columns_map, additional_gamma_test)

	# Set up correct model matrix
	for i in range(n):
		correct_model_matrix[i] = correct_model_tests[i + 1][n:] - correct_model_tests[i][n:]
	correct_model_matrix[n] = correct_model_tests[0][n:] - correct_model_tests[n][n:]
	# Set up incomplete model matrix
	for i in range(t):
		incomplete_model_matrix[i] = incomplete_model_tests[i + 1][t:] - incomplete_model_tests[i][t:]
	incomplete_model_matrix[t] = incomplete_model_tests[0][t:] - incomplete_model_tests[t][t:]
	pdb.set_trace()
	# Calculate new singular values
	correct_model_svs = np.linalg.svd(correct_model_matrix)[1]
	incomplete_model_svs = np.linalg.svd(incomplete_model_matrix)[1]

	# Classify the models
	correct_model_class = classify_model_as_incomplete(correct_model_svs)
	incomplete_model_class = classify_model_as_incomplete(incomplete_model_svs)
	# Add data
	correct_model["class"] = correct_model_class
	incomplete_model["class"] = incomplete_model_class

	return correct_model, incomplete_model


def test_incomplete_gamma_model(self, mRNA, miRNA):
	n = self.x + self.y
	t = mRNA + miRNA
	incomplete_model = {}

	# Decide which nodes to use

	rows, columns, rows_map, columns_map = self.build_incomplete_gamma_model(mRNA, miRNA)
	# Store data
	incomplete_model["rows"] = rows
	incomplete_model["columns"] = columns

	# Get visible test data
	visible_gamma_tests = get_sub_matrix(rows_map, columns_map, self.gamma_tests)
	incomplete_model["visible tests"] = visible_gamma_tests

	# Get visible matrix
	visible_gamma_matrix = np.zeros([t, t])
	for i in range(t - 1):
		visible_gamma_matrix[i] = visible_gamma_tests[i + 1][t:] - visible_gamma_tests[i][t:]
	visible_gamma_matrix[t - 1] = visible_gamma_tests[0][t:] - visible_gamma_tests[t - 1][t:]
	incomplete_model["visible matrix"] = visible_gamma_matrix

	# Get visible predictions
	vis_null, vis_svs = svd(visible_gamma_matrix)
	vis_null = vis_null[0]
	incomplete_model["visible nullspace"] = vis_null
	incomplete_model["visible svs"] = vis_svs

	# Check accuracy of incomplete model. If we used just this, how accuracte would it be?
	vis_real_vec = get_sub_matrix([0], columns_map, np.array([self.real_vector]))[0][t:]
	vis_acc = vector_estimate_accuracy(vis_null, vis_real_vec)
	incomplete_model["visible accuracy"] = vis_acc

	# Add a breaking test
	correct_model, more_incomplete_model_data = self.breaking_gamma_test(t, self.gamma_tests, visible_gamma_tests,
																		 columns_map)
	incomplete_model.update(more_incomplete_model_data)
	pdb.set_trace()

	return correct_model['class'], incomplete_model['class']


def build_overdetermined_lambda_model(self, mRNA, miRNA):
	actual_n = self.x + self.y
	new_nodes_total = mRNA + miRNA
	n = actual_n + new_nodes_total
	overdetermined_model = {}

	tests = np.zeros([n, n])
	temp_tests_1 = np.zeros([n, n])
	temp_tests_2 = np.zeros([n, n])
	new_wild_type_test = np.zeros([1, n])

	rows_to_change = range(self.x)
	rows_to_change.extend(range(self.x + mRNA, actual_n + mRNA))
	rows_map = dict(zip(range(actual_n), rows_to_change))
	columns_map = deepcopy(rows_map)

	set_sub_matrix(rows_map, columns_map, temp_tests_1, self.temp_lambda_tests_1)
	set_sub_matrix(rows_map, columns_map, temp_tests_2, self.temp_lambda_tests_2)

	changes = map(lambda x: x if x < self.x else x + mRNA, self.creation_rate_changes)
	overdetermined_model['changes'] = changes

	# Generate fake data. Gammas are superfluous
	ks, mus, gammas = generate_parameters(mRNA, miRNA)
	overdetermined_model["parameters"] = [ks, mus, gammas]

	new_rows = range(self.x, self.x + mRNA)
	new_rows.extend(range(actual_n + mRNA, n))

	for i in range(new_nodes_total):
		extra_node_mean = ks[i] / mus[i]
		current_new_RNA = new_rows[i]
		print(current_new_RNA)
		for j in range(n):
			temp_tests_1[j][current_new_RNA] = np.random.normal(extra_node_mean, extra_node_mean * .01)
			temp_tests_2[j][current_new_RNA] = np.random.normal(extra_node_mean, extra_node_mean * .01)
		# Add fake lambda modification
		multiplier = random.uniform(5, 10)
		temp_tests_1[current_new_RNA][current_new_RNA] = np.random.normal(extra_node_mean, extra_node_mean * .01)
		temp_tests_2[current_new_RNA][current_new_RNA] = np.random.normal(extra_node_mean, extra_node_mean * .01)
		new_wild_type_test[0][current_new_RNA] = extra_node_mean

		# Generate actual node data for the extra rows.
		for j in range(actual_n):
			new_value_1 = np.random.normal(self.wild_type_test[actual_n + j], self.wild_type_test[actual_n + j] * .01)
			if j == self.lambda_removed:
				new_value_2 = 0
			else:
				new_value_2 = np.random.normal(self.wild_type_test[actual_n + j],
											   self.wild_type_test[actual_n + j] * .01)
			if j < self.x:
				temp_tests_1[current_new_RNA][j] = new_value_1
				temp_tests_2[current_new_RNA][j] = new_value_2
			else:
				temp_tests_1[current_new_RNA][mRNA + j] = new_value_1
				temp_tests_2[current_new_RNA][mRNA + j] = new_value_2

	set_sub_matrix([0], columns_map, new_wild_type_test, np.array([self.wild_type_test[actual_n:]]))

	overdetermined_model["temp tests 1"] = temp_tests_1
	overdetermined_model["temp tests 2"] = temp_tests_2
	overdetermined_model['wild type test'] = new_wild_type_test[0]

	tests = temp_tests_1 - temp_tests_2
	overdetermined_model["tests"] = tests

	matrix = np.zeros([n, n])
	for i in range(n - 1):
		matrix[i] = tests[i + 1] - tests[i]
	overdetermined_model["matrix"] = matrix

	return overdetermined_model


def test_overdetermined_lambda_model(self, mRNA, miRNA):
	actual_n = self.x + self.y
	overdetermined_model = self.build_overdetermined_lambda_model(mRNA, miRNA)
	nullspace, svs = svd(overdetermined_model["matrix"])
	overdetermined_model["nullspace"] = nullspace
	overdetermined_model["svs"] = svs

	temp_tests_1 = overdetermined_model['temp tests 1']
	new_wild_type_test = overdetermined_model['wild type test']
	changes = overdetermined_model['changes']

	correct_model_nodes = classify_nodes_as_in_network(self.x, self.y, 0, 0, self.temp_lambda_tests_1,
													   self.wild_type_test[actual_n:], self.creation_rate_changes)
	overdetermined_model_nodes = classify_nodes_as_in_network(self.x, self.y, mRNA, miRNA, temp_tests_1,
															  new_wild_type_test, changes)
	pdb.set_trace()

	if correct_model_nodes[0]['True Positive'] == actual_n:
		correct_model_class = True
	else:
		correct_model_class = False

	if overdetermined_model_nodes[0]['True Positive'] == actual_n + mRNA + miRNA:
		overdetermined_model_class = True
	else:
		overdetermined_model_class = False

	overdetermined_model["class"] = overdetermined_model_class
	pdb.set_trace()

	return correct_model_class, overdetermined_model['class'], overdetermined_model_nodes


# def test_incomplete_knockout_lambda_model(self, mRNA, miRNA):
# 	n = self.x + self.y
# 	t = miRNA + mRNA
#
# 	# Build the visible matrix
# 	columns = self.build_incomplete_lambda_model(mRNA,miRNA)
# 	columns.extend(map(lambda x: x + n, columns))
#
# 	rows = deepcopy(columns)
# 	for i in range(t):
# 			rows[i] +=1
# 	for i in range(t,2*t):
# 		if rows[i] > self.lambda_removed + n:
# 			rows[i] -=1
# 	# Get rid of the extra knockout
# 	del rows[t-1]
# 	# Include a wild type
# 	rows.append(0)
# 	rows.remove(self.lambda_removed + n)
# 	rows.append(2*n-1)
# 	rows.sort()
#
# 	vis_matrix = get_sub_matrix(rows, columns, self.knockout_lambda_matrix)
#
# 	# Calculate values
# 	vis_null, vis_svs = svd(vis_matrix)
# 	vis_null = vis_null[0]
# 	vis_real_vec = get_sub_matrix([0], columns, np.array([self.real_vector]))[0]
#
# 	# Calculate visible accuracies
# 	vis_acc = vector_estimate_accuracy(vis_null, vis_real_vec)
# 	vis_k_acc = vector_estimate_accuracy(vis_null[:t], vis_real_vec[:t])
# 	vis_mu_acc = vector_estimate_accuracy(vis_null[t:], vis_real_vec[t:])
# 	vis_mean_acc = vector_estimate_accuracy(vis_null[:t]/vis_null[t:], vis_real_vec[:t]/vis_real_vec[t:])
#
# 	# Record this model
# 	incomplete_model = [(mRNA,miRNA), columns, vis_matrix, vis_null, vis_svs, [vis_acc, vis_k_acc, vis_mu_acc, vis_mean_acc]]
#
# 	# Test the prediction method
# 	additional_lambda_test, correct_model, incomplete_model_new_data = self.breaking_knockout_lambda_test(t,n, deepcopy(vis_matrix), deepcopy(self.knockout_lambda_matrix), columns)
#
# 	incomplete_model.extend(incomplete_model_new_data)

#	return additional_lambda_test, correct_model, incomplete_model



# def breaking_knockout_lambda_test(self, t, n, incomplete_model_matrix, correct_model_matrix, columns):
# 	# Data that will be returned
# 	correct_model = []
# 	incomplete_model = []
#
# 	# Choose which lambda value to change, an
# d create it
# 	if self.lambda_removed != columns[t-1]:
# 		change = columns[t-1]
# 	else:
# 		change = columns[0]
# 	new_lambda_value = self.ks[change] * np.random.uniform(.1,10)
#
# 	# Run another lambda test
# 	additional_lambda_test = np.zeros([1,2*n])
# 	temp1 = base_mean_field_solution(deepcopy(self.ks), self.mus, self.gammas, self.x, self.y, [], [], [(change, new_lambda_value)], None)
# 	temp2 = base_mean_field_solution(deepcopy(self.ks), self.mus, self.gammas, self.x, self.y, [self.lambda_removed], [], [(change, new_lambda_value)], None)
# 	additional_lambda_test[0][n:] = temp1 - temp2
# 	additional_lambda_test[0][self.lambda_removed] = 1
#
# 	# Insert the lambda test
# 	correct_model_matrix[2*n-1] = additional_lambda_test[0]
# 	incomplete_model_matrix[2*t-1] = get_sub_matrix([0], columns, additional_lambda_test)
# 	# Add to data
# 	correct_model.append(correct_model_matrix)
# 	incomplete_model.append(incomplete_model_matrix)
#
# 	# Calculate new singular values
# 	correct_model_svs = np.linalg.svd(correct_model_matrix)[1]
# 	incomplete_model_svs = np.linalg.svd(incomplete_model_matrix)[1]
# 	# Add to data
# 	correct_model.append(correct_model_svs)
# 	incomplete_model.append(incomplete_model_svs)
#
# 	# Classify
# 	correct_model_class = classify_model_as_incomplete(correct_model_svs)
# 	incomplete_model_class = classify_model_as_incomplete(incomplete_model_svs)
# 	# Add to data
# 	correct_model.append(correct_model_class)
# 	incomplete_model.append(incomplete_model_class)
#
# 	if correct_model_class != True:
# 		pdb.set_trace()
#
# 	#pdb.set_trace()
#
# 	return additional_lambda_test, correct_model, incomplete_model


def build_incomplete_gamma_model(self, mRNA, miRNA):
	n = self.x + self.y
	t = miRNA + mRNA
	rows = random.sample(range(n), t)
	rows.sort()
	columns = deepcopy(rows)
	columns.extend(map(lambda x: x + n, columns))
	return rows, columns


def breaking_gamma_test(self, t, old_correct_model_tests, old_incomplete_model_tests, columns):
	n = self.x + self.y

	# Data to return
	correct_model = {}
	incomplete_model = {}
	# New test data
	correct_model_tests = np.zeros([n + 1, 2 * n])
	incomplete_model_tests = np.zeros([t + 1, 2 * t])
	# New matrices that include new test
	correct_model_matrix = np.zeros([n + 1, n])
	incomplete_model_matrix = np.zeros([t + 1, t])

	# Run an additional test
	additional_gamma_test = np.ones([1, 2 * n])
	gamma_changes = []
	mRNA = random.choice(range(self.x))
	for miRNA in self.gammas[mRNA]:
		new_gamma_value = np.random.uniform(0.1, 10) * self.gammas[mRNA][miRNA]
		gamma_changes.append((mRNA, miRNA, new_gamma_value))
	additional_gamma_test[0][n:] = base_mean_field_solution(self.ks, self.mus, self.gammas, self.x, self.y, [],
															gamma_changes, [], deepcopy(self.sim))

	# Copy old tests
	set_sub_matrix(range(n), range(2 * n), correct_model_tests, old_correct_model_tests)
	set_sub_matrix(range(t), range(2 * t), incomplete_model_tests, old_incomplete_model_tests)
	# Incorporate the new test
	correct_model_tests[n] = additional_gamma_test
	incomplete_model_tests[t] = get_sub_matrix([0], columns, additional_gamma_test)
	# Add data
	correct_model["tests"] = correct_model_tests
	incomplete_model["tests"] = incomplete_model_tests

	# Set up correct model matrix
	for i in range(n):
		correct_model_matrix[i] = correct_model_tests[i + 1][n:] - correct_model_tests[i][n:]
	correct_model_matrix[n] = correct_model_tests[0][n:] - correct_model_tests[n][n:]
	# Set up incomplete model matrix
	for i in range(t):
		incomplete_model_matrix[i] = incomplete_model_tests[i + 1][t:] - incomplete_model_tests[i][t:]
	incomplete_model_matrix[t] = incomplete_model_tests[0][t:] - incomplete_model_tests[t][t:]
	# Add data
	correct_model["matrix"] = correct_model_matrix
	incomplete_model["matrix"] = incomplete_model_matrix

	# Calculate new singular values
	correct_model_null, correct_model_svs = svd(correct_model_matrix)
	incomplete_model_null, incomplete_model_svs = svd(incomplete_model_matrix)
	# Add data
	correct_model["svs"] = correct_model_svs
	incomplete_model["svs"] = incomplete_model_svs

	# Classify the models
	correct_model_class = classify_model_as_incomplete(correct_model_svs)
	incomplete_model_class = classify_model_as_incomplete(incomplete_model_svs)
	# Add data
	correct_model["class"] = correct_model_class
	incomplete_model["class"] = incomplete_model_class

	return correct_model, incomplete_model

	# Incomplete lambda test
	l_inc_data = current_network.test_incomplete_lambda_model(mRNA, miRNA - 2)
	correct_model_class = l_inc_data[0]["class"]
	if correct_model_class:
		l_inc_classifications["True Positive"] += 1
	else:
		l_inc_classifications["False Negative"] += 1
	if not l_inc_data[1]["class"]:
		l_inc_classifications["True Negative"] += 1
	else:
		l_inc_classifications["False Positive"] += 1
	# if l_inc_classifications["False Negative"] > 0 or l_inc_classifications["False Positive"] > 0:
	#	pdb.set_trace()

	# Knockout and double knockout accuracy


self.knockout_double_null_space, self.knockout_double_singular_values = svd(self.knockout_double_matrix)
self.knockout_double_prediction = abs(self.knockout_double_null_space[0])
self.knockout_double_accuracy[0] = vector_estimate_accuracy(self.knockout_double_prediction, self.real_vector)
self.knockout_double_accuracy[1] = vector_estimate_accuracy(self.knockout_double_prediction[:n], self.real_vector[:n])
self.knockout_double_accuracy[2] = vector_estimate_accuracy(self.knockout_double_prediction[n:], self.real_vector[n:])
self.knockout_double_accuracy[3] = vector_estimate_accuracy(
	self.knockout_double_prediction[:n] / self.knockout_double_prediction[n:],
	self.real_vector[:n] / self.real_vector[n:])


def double_knockout_test(self, method):
	tests = 1
	n = self.x + self.y
	testData = np.ones([n, 2 * n])

	alreadyUsed = [(-1, -1)]
	for i in range(n):
		print("Double Knockout Test: " + str(tests))
		tests += 1

		firstRNA = -1
		secondRNA = -1
		while (firstRNA, secondRNA) in alreadyUsed:
			firstRNA, secondRNA = random.sample(range(n), 2)
		alreadyUsed.append((firstRNA, secondRNA))
		self.double_knockout_removals.append((firstRNA, secondRNA))

		# Get a row
		testData[i][n:] = method(deepcopy(self.ks), self.mus, self.gammas, self.x, self.y, [firstRNA, secondRNA], [],
								 [], deepcopy(self.sim))
		testData[i][firstRNA] = 0
		testData[i][secondRNA] = 0
	return testData

	# Double knockout tests
	self.double_knockout_tests = np.zeros([n, 2 * n])
	self.double_knockout_removals = []

	# Knockout and Double Knockout Accuracy
	self.knockout_double_null_space = np.zeros(2 * n)
	self.knockout_double_prediction = np.zeros(2 * n)
	self.knockout_double_singular_values = np.zeros(2 * n)
	self.knockout_double_accuracy = np.zeros(4)
	self.knockout_double_mean_accuracy = np.zeros(n)

	if double_k_test:


print("Running double knockouts")
self.double_knockout_tests = self.double_knockout_test(method)  # Knockout and double knockout
self.knockout_double_matrix[0] = self.wild_type_test
set_sub_matrix(range(1, n + 1), range(2 * n), self.knockout_double_matrix, self.single_knockout_tests)
set_sub_matrix(range(n + 1, 2 * n), range(2 * n), self.knockout_double_matrix, self.double_knockout_tests)
self.knockout_gamma_matrix[2 * n - 1] = np.zeros(2 * n)


def breaking_lambda_test(self, t, old_correct_model_tests, old_incomplete_model_tests, columns, columns_map):
	n = self.x + self.y
	# Data to return
	correct_model = {}
	incomplete_model = {}
	# New test data
	correct_model_tests = np.zeros([n + 1, 2 * n])
	incomplete_model_tests = np.zeros([t + 1, 2 * t])
	# New matrices that include new test
	correct_model_matrix = np.zeros([n + 1, n])
	incomplete_model_matrix = np.zeros([t + 1, t])

	# Choose which lambda value to change, and create it
	if self.lambda_removed != columns[t - 1]:
		change = columns[t - 1]
	else:
		change = columns[0]
	new_lambda_value = self.ks[change] * np.random.uniform(.1, 10)

	# Run another lambda test
	additional_lambda_test = np.zeros([1, 2 * n])
	temp1 = base_mean_field_solution(deepcopy(self.ks), self.mus, self.gammas, self.x, self.y, [], [],
									 [(change, new_lambda_value)], None)
	temp2 = base_mean_field_solution(deepcopy(self.ks), self.mus, self.gammas, self.x, self.y, [self.lambda_removed],
									 [], [(change, new_lambda_value)], None)
	additional_lambda_test[0][n:] = temp1 - temp2
	additional_lambda_test[0][self.lambda_removed] = 1

	# Copy old tests
	set_sub_matrix(range(n), range(2 * n), correct_model_tests, old_correct_model_tests)
	set_sub_matrix(range(t), range(2 * t), incomplete_model_tests, old_incomplete_model_tests)
	# Incorporate the new test
	correct_model_tests[n] = additional_lambda_test
	incomplete_model_tests[t] = get_sub_matrix([0], columns_map, additional_lambda_test)
	correct_model["breaking tests"] = correct_model_tests
	incomplete_model["breaking tests"] = incomplete_model_tests

	# Set up correct model matrix
	for i in range(n):
		correct_model_matrix[i] = correct_model_tests[i + 1][n:] - correct_model_tests[i][n:]
	correct_model_matrix[n] = correct_model_tests[0][n:] - correct_model_tests[n][n:]
	# Set up incomplete model matrix
	for i in range(t):
		incomplete_model_matrix[i] = incomplete_model_tests[i + 1][t:] - incomplete_model_tests[i][t:]
	incomplete_model_matrix[t] = incomplete_model_tests[0][t:] - incomplete_model_tests[t][t:]
	# Add data
	correct_model["matrix"] = correct_model_matrix
	incomplete_model["matrix"] = incomplete_model_matrix

	# Calculate new singular values
	correct_model_svs = np.linalg.svd(correct_model_matrix)[1]
	incomplete_model_svs = np.linalg.svd(incomplete_model_matrix)[1]
	# Add data
	correct_model["svs"] = correct_model_svs
	incomplete_model["svs"] = incomplete_model_svs

	# Classify the models
	correct_model_class = classify_model_as_incomplete(correct_model_svs)
	incomplete_model_class = classify_model_as_incomplete(incomplete_model_svs)
	# Add data
	correct_model["class"] = correct_model_class
	incomplete_model["class"] = incomplete_model_class

	return correct_model, incomplete_model


def test_incomplete_lambda_model(self, mRNA, miRNA):
	n = self.x + self.y
	t = mRNA + miRNA
	incomplete_model = {}

	rows, columns, rows_map, columns_map = self.build_incomplete_lambda_model(mRNA, miRNA)
	incomplete_model["rows"] = rows
	incomplete_model["columns"] = columns
	# pdb.set_trace()
	visible_lambda_tests = get_sub_matrix(rows_map, columns_map, self.lambda_tests)
	incomplete_model["visible tests"] = visible_lambda_tests

	visible_lambda_matrix = np.zeros([t, t])
	for i in range(t - 1):
		visible_lambda_matrix[i] = visible_lambda_tests[i + 1][t:] - visible_lambda_tests[i][t:]
	incomplete_model["visible matrix"] = visible_lambda_matrix

	vis_null, vis_svs = svd(visible_lambda_matrix)
	vis_null = vis_null[0]
	vis_est = abs(vis_null)
	# pdb.set_trace()
	incomplete_model["visible nullspace"] = vis_null
	incomplete_model["visible svs"] = vis_svs
	incomplete_model["visible estimate"] = vis_est
	# print(vis_null)
	# print(classify_small_model_as_incomplete(mRNA, miRNA, vis_null))

	# Check accuracy of incomplete model. If we used just this, how accuracte would it be?
	vis_real_vec = get_sub_matrix([0], columns_map, np.array([self.real_vector]))[0][t:]
	vis_acc = vector_estimate_accuracy(vis_est, vis_real_vec)
	incomplete_model["visible accuracy"] = vis_acc

	# correct_model_class = classify_half_nullspace_as_incorrect(self.x, self.y, self.lambda_nullspace[0])
	# incomplete_model_class = classify_half_nullspace_as_incorrect(mRNA, miRNA, vis_null)
	# incomplete_model["class"] = incomplete_model_class

	correct_model, more_incomplete_model_data = self.breaking_lambda_test(t, self.lambda_tests, visible_lambda_tests,
																		  columns, columns_map)
	incomplete_model.update(more_incomplete_model_data)

	return correct_model, incomplete_model


def build_incomplete_lambda_model(self, mRNA, miRNA):
	n = self.x + self.y
	t = miRNA + mRNA

	mRNAs_to_choose = set(range(self.x))
	miRNAs_to_choose = set(range(self.x, n))
	if self.lambda_removed < self.x:
		mRNAs_to_choose.remove(self.lambda_removed)
		columns = random.sample(mRNAs_to_choose, mRNA - 1)
		columns.extend(random.sample(miRNAs_to_choose, miRNA))
	else:
		miRNAs_to_choose.remove(self.lambda_removed)
		columns = random.sample(mRNAs_to_choose, mRNA)
		columns.extend(random.sample(miRNAs_to_choose, miRNA - 1))
	columns.append(self.lambda_removed)

	columns.sort()
	rows = deepcopy(columns)

	columns.extend(map(lambda x: x + n, columns))

	rows_map = {}
	columns_map = {}

	for i in range(len(rows)):
		rows_map[rows[i]] = i
	for i in range(len(columns)):
		columns_map[columns[i]] = i

	return rows, columns, rows_map, columns_map
