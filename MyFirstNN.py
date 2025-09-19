import numpy as np

# Learning rate decay code

np.random.seed(0)


# Our Sample Data set / It's a bunch of ordered random numbers put into an array of n by k / The data is fake and made up
def create_data(n, k):

    X = np.zeros((n*k, 2))  # data matrix (each row = single example)
    y = np.zeros(n*k, dtype='uint8')  # class labels
    for j in range(k):
        ix = range(n*j, n*(j+1))
        r = np.linspace(0.0, 1, n)  # radius
        t = np.linspace(j*4, (j+1)*4, n) + np.random.randn(n)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = j

    return X, y


# Dense Layer / Layer of neurons and their attributes / This is depicting a fully connected neuron layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, inputs, neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))

    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        # Calculate output values from input ones, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dvalues = np.dot(dvalues, self.weights.T)


# ReLU activation / This activates neurons that meet certain criteria and deactivate ones that don't
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # We need to modify original values, we're copying the data as to not overwrite it immediately before all calculations
        self.dvalues = dvalues.copy()

        # Zero gradient where input values were negative
        self.dvalues[self.inputs <= 0] = 0

# Softmax activation / This takes output data and squishes the numbers to manageable sizes and also normalizes the probabilities
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):
        # get un-normalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):
        self.dvalues = dvalues.copy()


# This calculates how wrong the predictions are by comparing to what they should be
# Once again the data in this code are made up and random, and so are the labels (y in the data set function)
class Loss_CategoricalCrossEntropy:

    def forward(self, y_pred, y_true):

        # Number of samples in batches
        samples = len(y_pred)

        # Probabilities for target values
        y_pred = y_pred[range(samples), y_true]

        # Losses
        negative_log_likelihoods = -np.log(y_pred)

        # Overall Loss
        data_loss = np.mean(negative_log_likelihoods)

        return data_loss

    # Backwards pass
    def backward(self, dvalues, y_true):

        samples = dvalues.shape[0]

        self.dvalues = dvalues.copy() # need to make a copy
        self.dvalues[range(samples), y_true] -= 1
        self.dvalues = self.dvalues / samples


# This allows for you to update weights and biases of all the neurons, controls by how much they change, and help the
# backpropogation from getting stuck in a local minimum (SGD = Stochastic Gradient Descent)
class Optimizer_SGD:

    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1., decay=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    # Call once before any param updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.current_learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

    # Call post updates
    def post_update_params(self):
        self.iterations += 1


# This function is just the main function of the program I was just too lazy to actually write a main function and break all this
# code up
def AI_with_learning_rate_decay():
    # Create dataset
    X, y = create_data(100, 3) # That's a 100 row by 3 column array  -  X = the first two columns, y = 3rd column

    # Create Dense layer with 2 input features and 64 output values
    dense1 = Layer_Dense(2, 64)  # first dense layer, 2 inputs (each sample has 2 features), 64 outputs

    # Create ReLU activation (to be used with Dense layer):
    activation1 = Activation_ReLU()

    # Create second Dense layer with 64 input features and 3 output values (output values)
    dense2 = Layer_Dense(64, 3)  # second dense layer, 64 inputs, 3 outputs

    # Create Softmax activation (to be used with Dense layer) this is the final output so we need to make the numbers manageable
    activation2 = Activation_Softmax()

    # Create loss function (how wrong were the predictions)
    loss_function = Loss_CategoricalCrossEntropy()

    # Create optimizer (Uses basic Stochastic Gradient Descent AKA finding global minimum of function using derivatives)
    optimizer = Optimizer_SGD(decay=1e-6)

    # Train in loop
    for epoch in range(10001):
        # Make a forward pass of our training data thru this layer
        dense1.forward(X)

        # Make a forward pass thru activation function - we take output of previous layer here
        activation1.forward(dense1.output)

        # Make a forward pass thru second Dense layer - it takes outputs of activation function of first layer as inputs
        dense2.forward(activation1.output)

        # Make a forward pass thru activation function - we take output of previous layer here
        activation2.forward(dense2.output)

        # Calculate loss from output of activation2 so softmax activation
        loss = loss_function.forward(activation2.output, y)

        # Calculate accuracy from output of activation2 and targets
        predictions = np.argmax(activation2.output, axis=1)  # calculate values along first axis
        accuracy = np.mean(predictions == y)

        if not epoch % 100:
            print(f'epoch: {epoch}, accuracy: {accuracy:.3f}, loss: {loss:.3f}, learning rate: {optimizer.current_learning_rate}')

        # Backward pass
        loss_function.backward(activation2.output, y)
        activation2.backward(loss_function.dvalues)
        dense2.backward(activation2.dvalues)
        activation1.backward(dense2.dvalues)
        dense1.backward(activation1.dvalues)

        # Update weights
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

    return


AI_with_learning_rate_decay()
