#!/usr/bin/env python
import numpy as np
def dsigmoid(y):
    return 1-y**2
ds = np.vectorize(dsigmoid)
train_in = [[0, 0], [0, 1], [1, 0], [1, 1]]
train_out = [0, 1, 1, 0]
[x.append(1) for x in train_in]
d_input = len(train_in[0])
d_hidden = 16
d_output = 1
init_wt = 0.02
input_weights = np.random.normal(0.0, init_wt, size=(d_input, d_hidden))
output_weights = np.random.normal(0.0, init_wt, size=(d_hidden, d_output))
learning_rate = 0.3
epochs = 1000
for e in range(epochs):
	pred = []
	for m in range(len(train_in)):
		o_i = np.tanh(np.dot(input_weights.T, train_in[m]))
		o_h = np.ravel(np.tanh(np.dot(output_weights.T, o_i))) #Output of network
		o_d = np.multiply(np.ravel(np.subtract(train_out[m], o_h)), ds(o_h))
		h_d = np.multiply(ds(o_i), np.dot(o_d, output_weights.T))
		o_i = np.ravel(o_i)
		output_weights = np.add(np.multiply(learning_rate, np.multiply(o_d, o_i)).reshape(d_hidden, 1), output_weights)
		input_weights = np.add(input_weights, np.multiply(learning_rate, np.multiply(h_d.reshape(d_hidden, 1), train_in[m]).T))