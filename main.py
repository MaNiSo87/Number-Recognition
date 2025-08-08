import numpy as np
from datasets import load_dataset
import random
from illustration import illustration as ill
dataset = load_dataset('mnist')

train = np.array(dataset['train']['image']) / 255.0
train_label = dataset['train']['label']
test = np.array(dataset['test']['image']) / 255.0
test_label = dataset['test']['label']




class Agent():
    def __init__(self):
        self.epochs = 1
        self.lr = 0.01

        self.layer_one = (28 * 28)
        self.layer_two = 128
        self.output_layer = 10

        self.layer_one_w = np.array([[random.uniform(-0.1, 0.1) for _ in range(self.layer_one)] for _ in range(self.layer_two)])
        self.layer_two_w = np.array([[random.uniform(-0.1, 0.1) for _ in range(self.layer_two)] for _ in range(self.output_layer)])
        self.layer_one_b = np.zeros(self.layer_two)
        self.layer_two_b = np.zeros(self.output_layer)
    

    def ReLU(self, z1):
        return np.maximum(0, z1)

    def softmax(self, z2):
        exp_z2 = np.exp(z2 - np.max(z2))
        return exp_z2 / exp_z2.sum()

    def cross_entropy(self, y_true, output):
        return -np.sum(y_true * np.log(output + 1e-8)) 

    def eval(self, test, test_label):
        correct = 0
        for j, pic in enumerate(test):
            pic = pic.flatten()
            z1 = np.dot(self.layer_one_w, pic.T).flatten() + self.layer_one_b
            h1 = self.ReLU(z1)
            z2 = np.dot(self.layer_two_w, h1) + self.layer_two_b
            output = self.softmax(z2)
            pred = np.argmax(output)
            if pred == test_label[j]:
                correct += 1
        return correct / len(test) * 100


    def train(self, train, train_label):
        for i in range(self.epochs):
            indices = np.arange(len(train))
            np.random.shuffle(indices)
            train = train[indices]
            train_label = np.array(train_label)[indices]

            for j, pic in enumerate(train):
                pic = pic.flatten()

                z1 = np.dot(self.layer_one_w, pic.T).flatten() + self.layer_one_b
                h1 = self.ReLU(z1)

                z2 = np.dot(self.layer_two_w, h1) + self.layer_two_b
                output = self.softmax(z2)

                y_true = np.zeros(self.output_layer)
                y_true[train_label[j]] += 1

                loss = self.cross_entropy(y_true, output)

                d_z2 = output - y_true
                d_layer_two_w = np.outer(d_z2, h1) 
                d_layer_two_b = d_z2 
                d_h1 = np.dot(self.layer_two_w.T, d_z2) 
                d_z1 = d_h1 * (z1 > 0) 
                d_layer_one_w = np.outer(d_z1, pic) 
                d_layer_one_b = d_z1
                
                self.layer_one_w -= self.lr * d_layer_one_w
                self.layer_one_b -= self.lr * d_layer_one_b
                self.layer_two_w -= self.lr * d_layer_two_w
                self.layer_two_b -= self.lr * d_layer_two_b

                if j % 1000 == 0:
                    acc = self.eval(test, test_label)
                    print(f'Epoch: {i} | Data_Num: {j} | Loss: {loss} | Accuracy: {acc}')


    def test(self, test, test_label):
        for j, pic in enumerate(test):
            pic = pic.flatten()

            z1 = np.dot(self.layer_one_w, pic.T).flatten() + self.layer_one_b
            h1 = self.ReLU(z1)

            z2 = np.dot(self.layer_two_w, h1) + self.layer_two_b
            output =self.softmax(z2)

            pred = np.argmax(output)

            print(f'Label: {test_label[j]} | Pred: {pred}')
            ill(dataset['test']['image'][j])








agent = Agent()
agent.train(train, train_label)
agent.test(test[:200], test_label[:200])