# Following the excellent blog post 
# http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
import pandas as pd
import numpy as np
import nltk
import itertools
import operator
from datetime import datetime

vocab_size = 8000
unknown_token = "UNKNOWN_TOKEN"
start_token = "SENTENCE_START_TOKEN"
end_token = "SENTENCE_END_TOKEN"

train_data = pd.read_csv('20190928-reviews.csv')
train_data = train_data["body"]
train_data = train_data.values[:1000] # Limit data to 1000 for faster speed
sentences = []
for i in range(len(train_data)):
    if (hasattr(train_data[i], 'lower')):
        new_sentences = nltk.sent_tokenize(train_data[i])
        for sentence in new_sentences:
            sentences.append(start_token + " " + sentence + " " + end_token)
#train_data = [nltk.sent_tokenize(x) for x in train_data]

tokenized_sentences = [nltk.word_tokenize(x) for x in sentences]
print (tokenized_sentences[:2])

word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
vocab = word_freq.most_common(vocab_size - 1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

for i, sentence in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sentence]

x_train = np.asarray([[word_to_index[w] for w in s[:-1]] for s in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in s[1:]] for s in tokenized_sentences])

class RNN:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        T = len(x)
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        o = np.zeros((T, self.word_dim))

        for t in np.arange(T):
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))
            o[t] = self.softmax(self.V.dot(s[t]))
        
        return [o, s]
    
    def softmax(self, vector):
        return np.exp(vector) / np.sum(np.exp(vector), axis=0)

    def predict(self, x):
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)
    
    def calculate_total_loss(self, x, y):
        loss = 0
        for i in np.arange(len(y)):
            [o, s] = self.forward_propagation(x[i])
            correct_predictions = o[np.arange(len(y[i])), y[i]]
            loss = loss + -1 * np.sum(np.log(correct_predictions))
        
        return loss

    def calculate_loss(self, x, y):
        n = np.sum([len(y_i) for y_i in y])
        return self.calculate_total_loss(x, y) / n

    def bptt(self, x, y):
        T = len(y)
        [o, s] = self.forward_propagation(x)
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] = delta_o[np.arange(len(y)), y] - 1
        
        for t in np.arange(T)[::-1]:
            dLdV = dLdV + np.outer(delta_o[t], s[t].T)
            delta_t = self.V.T.dot(delta_o[t])*(1-(s[t]**2))
            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t+1)[::-1]:
                dLdW = dLdW + np.outer(delta_t, s[bptt_step - 1])
                dLdU[:, x[bptt_step]] = dLdU[:, x[bptt_step]] + delta_t
                delta_t = self.W.T.dot(delta_t)*(1-(s[bptt_step-1]**2))
        
        return [dLdU, dLdV, dLdW]

    def sgd_step(self, x, y, learning_rate):
        [dLdU, dLdV, dLdW] = self.bptt(x, y)
        self.U = self.U - learning_rate * dLdU
        self.V = self.V - learning_rate * dLdV
        self.W = self.W - learning_rate * dLdW
    
    def train(self, x, y, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
            if (epoch % evaluate_loss_after == 0):
                loss = model.calculate_loss(x, y)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print ("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
            
            for i in range(len(y)):
                self.sgd_step(x[i], y[i], learning_rate)
                num_examples_seen = num_examples_seen + 1
    
    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        bptt_gradients = self.bptt(x, y)
        model_parameters = ['U', 'V', 'W']
        for pidx, pname in enumerate(model_parameters):
            parameter = operator.attrgetter(pname)(self)
            print ("Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape)))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), …
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                original_value = parameter[ix]

                # Estimate the gradient using (f(x+h) – f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x],[y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x],[y])
                estimated_gradient = (gradplus - gradminus)/(2*h)

                # Reset parameter to original value
                parameter[ix] = original_value
                backprop_gradient = bptt_gradients[pidx][ix]
                
                # calculate The relative error: (|x – y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
                if relative_error > error_threshold:
                    print ("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
                    print ("+h Loss: %f" % gradplus)
                    print ("-h Loss: %f" % gradminus)
                    print ("Estimated_gradient: %f" % estimated_gradient)
                    print ("Backpropagation gradient: %f" % backprop_gradient)
                    print ("Relative Error: %f" % relative_error)
                    return
                it.iternext()
            print ("Gradient check for parameter %s passed." % (pname))

# model = RNN(vocab_size)
# o, s = model.forward_propagation(x_train[0])
# print (o)

# grad_check_vocab_size = 100
# np.random.seed(10)
# model = RNN(grad_check_vocab_size, 10, bptt_truncate=1000)
# model.gradient_check([0,1,2,3], [1,2,3,4])

model = RNN(vocab_size)
loss = model.calculate_loss(x_train[:1000], y_train[:1000])
print (loss)
model.train(x_train[:5], y_train[:5])