#!/usr/bin/python3
import numpy as np
import math

class AdaBoost:
    def __init__(self, training_data):
        self.training_data = training_data
        self.n = len(self.training_data)
        self.weights = np.ones(self.n) / self.n
        self.alphas = []
        self.rules = []
    
    def set_rule(self, func, test=False):
        print('In set rule')
        print('training data =')
        print(self.training_data)
        errors = np.array([item[1] != func(item[0]) for item in self.training_data])
        for item in self.training_data:
            print('    item 0 = {} and item 1 = {}'.format(item[0], item[1]))
            print('        func item 0 = {}'.format(func(item[0])))

        print('errors =')
        print(errors)
        print('weights = ')
        print(self.weights)
        e = (errors * self.weights).sum()
        print('e = {}'.format(e))
        if test:
            return e
        
        # weight update
        alpha = 0.5 * np.log((1 - e) / e)
        print('e = {} alpha = {}'.format(e, alpha))
        w = np.zeros(self.n)
        for i in range(self.n):
            if errors[i]:
                w[i] = self.weights[i] * np.exp(alpha)
            else:
                w[i] = self.weights[i] * np.exp(-alpha)
        
        print('Updated weights = ')
        print(w)

        self.weights = w / w.sum()
        self.alphas.append(alpha)
        self.rules.append(func)
    
    def evaluate(self):
        nr = len(self.rules)
        print('self rules = ')
        print(self.rules)
        print('self alphas = ')
        print(self.alphas)
        for data, label in self.training_data:
            hx = [alpha * rule(data) for alpha, rule in zip(self.alphas, self.rules)]
            for i in range(nr):
                print('the alphas = {} and the rules = {} and rules data = {}'.format(self.alphas[i], self.rules[i], self.rules[i](data)))
            print('data = {} and sign = {}'.format(data, np.sign(label) == np.sign(sum(hx))))

if __name__ == '__main__':
    examples = []
    examples.append(((1, 2), 1))
    examples.append(((1, 4), 1))
    examples.append(((2.5,5.5), 1))
    examples.append(((2, 1), -1))
    examples.append(((5, 2), -1))

    m = AdaBoost(examples)
    m.set_rule(lambda x: 2 * (x[0] < 1.5) - 1)
    m.set_rule(lambda x: 2 * (x[0] < 4.5) - 1)
    m.evaluate()