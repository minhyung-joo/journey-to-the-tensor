# Following the excellent blog post 
# http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
import pandas as pd
import numpy as np
import nltk
import itertools

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

model = RNN(vocab_size)
o, s = model.forward_propagation(x_train[0])
print (o)