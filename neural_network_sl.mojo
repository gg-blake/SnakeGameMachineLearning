import numpy as np
import random
from utils import RGB

struct SingleLayerNeuralNetwork:
    var _count: Int
    def __init__(self, input_nodes, hidden_nodes, output_nodes, _count: Int, mother=RGB(-1,-1,-1), father=RGB(-1,-1,-1)):
        # Instance Variables #
        self.input_nodes = input_nodes # Number of nodes in input layer
        self.hidden_nodes = hidden_nodes # Number of nodes in hidden layer
        self.output_nodes = output_nodes # Number of nodes in output layer
        if mother.r == Int(-1):
            self.id = RGB(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            self.id = random.choice([mother, father])
        self.mother = mother # SingleLayerNeuralNetwork id of mother
        self.father = father # SingleLayerNeuralNetwork id of father
        self.weights_ih = np.random.rand(self.hidden_nodes, self.input_nodes) # Weights from input layer to hidden layer
        self.weights_ho = np.random.rand(self.output_nodes, self.hidden_nodes) # Weights from hidden layer to output layer
        self.bias_h = np.random.rand(self.hidden_nodes, 1) # Biases of hidden layer
        self.bias_o = np.random.rand(self.output_nodes, 1) # Biases of output layer

        # Instance Callback #
        SingleLayerNeuralNetwork._count += 1 # Increment total number of SingleLayerNeuralNetwork objects by 1

    # String representation
    def __str__(self):
        return f"weights_ih: {self.weights_ih}, weights_ho: {self.weights_ho}, bias_h: {self.bias_h}, bias_o: {self.bias_o}"
    
    # Printable representation
    def __repr__(self):
        return self.__str__()
    
    # Dictionary representation
    def __dict__(self):
        # This function enables models to be saved as JSON files
        return {
            "input_nodes": self.input_nodes,
            "hidden_nodes": self.hidden_nodes,
            "output_nodes": self.output_nodes,
            "weights_ih": self.weights_ih.tolist(),
            "weights_ho": self.weights_ho.tolist(),
            "bias_h": self.bias_h.tolist(),
            "bias_o": self.bias_o.tolist()
        }

    # Perform a feedforward operation on this SingleLayerNeuralNetwork object and return the output
    def feed(self, input_array):
        # Convert input array to numpy array
        inputs = np.array(input_array, ndmin=2).T

        # Feed input layer to hidden layer
        hidden = np.dot(self.weights_ih, inputs)
        hidden = np.add(hidden, self.bias_h)
        hidden = self.sigmoid(hidden)

        # Feed hidden layer to output layer
        output = np.dot(self.weights_ho, hidden)
        output = np.add(output, self.bias_o)
        output = self.sigmoid(output)

        return output
    
    # Determines the activation of a node
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Potentially mutate the weights and biases of this SingleLayerNeuralNetwork object
    def mutate(self, mutation_rate):
        # Vectorized function to mutate a value
        def mutate(val):
            # Mutate value if random number is less than mutation rate
            if random.random() < mutation_rate:
                if self.id:
                    # Keep the mutated id colors within the acceptable colorspace
                    self.id.r += random.randint(-20, 20)
                    self.id.g += random.randint(-20, 20)
                    self.id.b += random.randint(-20, 20)

                    if self.id.r < 0:
                        self.id.r = 0
                    elif self.id.r > 255:
                        self.id.r = 255

                    if self.id.g < 0:
                        self.id.g = 0
                    elif self.id.g > 255:
                        self.id.g = 0

                    if self.id.b < 0:
                        self.id.b = 0
                    elif self.id.b > 255:
                        self.id.b = 255

                    


                    
                    
                
                return val + random.gauss(0, 0.1)
            else:
                return val
            
        # Vectorize the mutate function
        mutate = np.vectorize(mutate)

        # Set weights and biases to new mutated numpy array
        self.weights_ih = mutate(self.weights_ih)
        self.weights_ho = mutate(self.weights_ho)
        self.bias_h = mutate(self.bias_h)
        self.bias_o = mutate(self.bias_o)


    # Returns a copy of this SingleLayerNeuralNetwork object
    def clone(self):
        # Create new SingleLayerNeuralNetwork object with identical instance variables
        nn = SingleLayerNeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes)
        # Copy the weights and biases
        nn.weights_ih = np.copy(self.weights_ih)
        nn.weights_ho = np.copy(self.weights_ho)
        nn.bias_h = np.copy(self.bias_h)
        nn.bias_o = np.copy(self.bias_o)
        nn.mother, nn.father = self.id, self.id
        return nn