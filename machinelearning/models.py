import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        value = nn.as_scalar(self.run(x))
        if value >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        while True:
            flag = True
            for x, y in dataset.iterate_once(1):
                value = self.get_prediction(x)
                value_ = nn.as_scalar(y)
                if (value == value_):
                    continue
                else:
                    self.w.update(x, value_)
                    flag = False
            if flag:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        self.layer_size = 20
        self.layer_number = 3
        #self.batch = 200
        self.learning_rate = 0.2

        self.m = []
        self.b = []

        self.m.append(nn.Parameter(1, self.layer_size))
        self.b.append(nn.Parameter(1, self.layer_size))

        for i in range(1, self.layer_number - 1):
            self.m.append(nn.Parameter(self.layer_size, self.layer_size))
            self.b.append(nn.Parameter(1, self.layer_size))

        self.m.append(nn.Parameter(self.layer_size, 1))
        self.b.append(nn.Parameter(1, 1)) 

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        for i in range(self.layer_number - 1):
            xi = nn.Linear(x, self.m[i])
            yi = nn.AddBias(xi, self.b[i])
            x = nn.ReLU(yi)

        xm = nn.Linear(x, self.m[self.layer_number - 1])
        predict_y = nn.AddBias(xm, self.b[self.layer_number - 1])
        return predict_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        predict_y = self.run(x)
        return nn.SquareLoss(predict_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        l = dataset.x.shape[0]
        ave_loss = float('inf')

        for x, y in dataset.iterate_forever(l):
            loss = self.get_loss(x, y)
            ave_loss = nn.as_scalar(loss)
            print(ave_loss)
            if ave_loss < 0.01:
                break
            grad_wrt = nn.gradients(loss, self.m + self.b)
            for i in range(self.layer_number):
                self.m[i].update(grad_wrt[i], -self.learning_rate)
                self.b[i].update(grad_wrt[i + self.layer_number], -self.learning_rate)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        self.layer_size = 200
        self.layer_number = 4
        self.batch_size = 500
        self.learning_rate = 0.5

        self.input_size = 784
        self.output_size = 10

        self.m = []
        self.b = []

        self.m.append(nn.Parameter(self.input_size, self.layer_size))
        self.b.append(nn.Parameter(1, self.layer_size))

        for i in range(1, self.layer_number - 1):
            self.m.append(nn.Parameter(self.layer_size, self.layer_size))
            self.b.append(nn.Parameter(1, self.layer_size))

        self.m.append(nn.Parameter(self.layer_size, self.output_size))
        self.b.append(nn.Parameter(1, self.output_size)) 

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        
        for i in range(self.layer_number - 1):
            xi = nn.Linear(x, self.m[i])
            yi = nn.AddBias(xi, self.b[i])
            x = nn.ReLU(yi)

        xm = nn.Linear(x, self.m[self.layer_number - 1])
        predict_y = nn.AddBias(xm, self.b[self.layer_number - 1])
        return predict_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        predict_y = self.run(x)
        return nn.SoftmaxLoss(predict_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        for x, y in dataset.iterate_forever(self.batch_size):
            loss = self.get_loss(x, y)
            grad_wrt = nn.gradients(loss, self.m + self.b)
            for i in range(self.layer_number):
                self.m[i].update(grad_wrt[i], -self.learning_rate)
                self.b[i].update(grad_wrt[i + self.layer_number], -self.learning_rate)
            accuracy = dataset.get_validation_accuracy()
            #print(accuracy)
            if accuracy >= 0.975:
                break

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        self.layer_size = 400
        self.batch_size = 10
        self.learning_rate = 0.2

        self.w = nn.Parameter(self.num_chars, self.layer_size)
        self.b = nn.Parameter(1, self.layer_size)

        self.w_hidden = nn.Parameter(self.layer_size, self.layer_size)
        self.b_hidden = nn.Parameter(1, self.layer_size)

        self.w_final = nn.Parameter(self.layer_size, 5)
        self.b_final = nn.Parameter(1, 5)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        #deal with xs[0]
        h = nn.Linear(xs[0], self.w)
        #deal with xs[1:]
        for i in range(1, len(xs)):
            z = nn.Add(nn.Linear(xs[i], self.w), nn.Linear(h, self.w_hidden))
            h = nn.AddBias(nn.AddBias(z, self.b), self.b_hidden)
            h = nn.ReLU(h)
        #deal with the output after xs
        h = nn.AddBias(nn.Linear(h, self.w_final), self.b_final)
        return h

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        predict_y = self.run(xs)
        return nn.SoftmaxLoss(predict_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        for x, y in dataset.iterate_forever(self.batch_size):
            loss = self.get_loss(x, y)
            parameters = [self.w, self.b, self.w_hidden, self.b_hidden, self.w_final, self.b_final]
            grad_wrt = nn.gradients(loss, parameters)
            for i in range(len(parameters)):
                parameters[i].update(grad_wrt[i], -self.learning_rate)
            accuracy = dataset.get_validation_accuracy()
            #print(accuracy)
            if accuracy >= 0.85:
                break
