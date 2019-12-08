import numpy as np
import re
import matplotlib.pyplot as plt

def tokenize(text):
    # obtains tokens with a least 1 alphabet
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())

def mapping(tokens):
    id_to_word = dict()
    word_to_id = dict()

    for i, w in enumerate(tokens):
        id_to_word[i] = w
        word_to_id[w] = i
    
    return [id_to_word, word_to_id]

def generate_training_data(tokens, word_to_id, window_size):
    N = len(tokens)
    x, y = [], []

    for i in range(N):
        nbr_inds = list(range(max(0, i - window_size), i)) + \
                    list(range(i + 1, min(N, i + window_size + 1)))

        for j in nbr_inds:
            x.append(word_to_id[tokens[i]])
            y.append(word_to_id[tokens[j]])
    
    x = np.array(x)
    x = np.expand_dims(x, axis=0)
    y = np.array(y)
    y = np.expand_dims(y, axis=0)
    
    return [x, y]

def initialize_wrd_emb(vocab_size, emb_size):
    WRD_EMB = np.random.randn(vocab_size, emb_size) * 0.01
    return WRD_EMB

def initialize_dense(input_size, output_size):
    W = np.random.randn(output_size, input_size) * 0.01
    return W

def initialize_parameters(vocab_size, emb_size):
    WRD_EMB = initialize_wrd_emb(vocab_size, emb_size)
    W = initialize_dense(emb_size, vocab_size)

    parameters = {}
    parameters["WRD_EMB"] = WRD_EMB
    parameters["W"] = W

    return parameters

def ind_to_word_vecs(inds, parameters):
    m = inds.shape[1]
    WRD_EMB = parameters['WRD_EMB']
    word_vec = WRD_EMB[inds.flatten(), :].T

    assert (word_vec.shape == (WRD_EMB.shape[1], m))

    return word_vec

def linear_dense(word_vec, parameters):
    m = word_vec.shape[1]
    W = parameters['W']
    Z = np.dot(W, word_vec)

    assert (Z.shape == (W.shape[0], m))

    return W, Z

def softmax(Z):
    softmax_out = np.divide(np.exp(Z), np.sum(np.exp(Z), axis=0, keepdims=True) + 0.001)

    assert (softmax_out.shape == Z.shape)

    return softmax_out

def forward_propagation(inds, parameters):
    word_vec = ind_to_word_vecs(inds, parameters)
    W, Z = linear_dense(word_vec, parameters)
    softmax_out = softmax(Z)

    caches = {}
    caches['inds'] = inds
    caches['word_vec'] = word_vec
    caches['W'] = W
    caches['Z'] = Z
    
    return softmax_out, caches

def cross_entropy(softmax_out, y):
    m = softmax_out.shape[1]
    cost = -(1/m)*np.sum(np.sum(y * np.log(softmax_out + 0.001), axis=0, keepdims=True), axis=1)
    return cost

def softmax_backward(y, softmax_out):
    dLdZ = softmax_out - y
    return dLdZ

def dense_backward(dLdZ, caches):
    W = caches['W']
    word_vec = caches['word_vec']
    m = word_vec.shape[1]

    dLdW = (1/m)*np.dot(dLdZ, word_vec.T)
    dLdWordVec = np.dot(W.T, dLdZ)

    return dLdW, dLdWordVec

def backpropagation(y, softmax_out, caches):
    dLdZ = softmax_backward(y, softmax_out)
    dLdW, dLdWordVec = dense_backward(dLdZ, caches)

    gradients = dict()
    gradients['dLdZ'] = dLdZ
    gradients['dLdW'] = dLdW
    gradients['dLdWordVec'] = dLdWordVec
    
    return gradients

def update_parameters(parameters, caches, gradients, learning_rate):
    vocab_size, emb_size = parameters["WRD_EMB"].shape
    inds = caches['inds']
    WRD_EMB = parameters['WRD_EMB']
    dL_dword_vec = gradients['dLdWordVec']
    m = inds.shape[-1]
    
    WRD_EMB[inds.flatten(), :] -= dL_dword_vec.T * learning_rate

    parameters['W'] -= learning_rate * gradients['dLdW']

def skipgram_model_training(X, Y, vocab_size, emb_size, learning_rate, epochs, batch_size=256, parameters=None, print_cost=True, plot_cost=True):
    """
    X: Input word indices. shape: (1, m)
    Y: One-hot encodeing of output word indices. shape: (vocab_size, m)
    vocab_size: vocabulary size of your corpus or training data
    emb_size: word embedding size. How many dimensions to represent each vocabulary
    learning_rate: alaph in the weight update formula
    epochs: how many epochs to train the model
    batch_size: size of mini batch
    parameters: pre-trained or pre-initialized parameters
    print_cost: whether or not to print costs during the training process
    """
    costs = []
    m = X.shape[1]
    
    if parameters is None:
        parameters = initialize_parameters(vocab_size, emb_size)
    
    for epoch in range(epochs):
        epoch_cost = 0
        batch_inds = list(range(0, m, batch_size))
        np.random.shuffle(batch_inds)
        for i in batch_inds:
            X_batch = X[:, i:i+batch_size]
            Y_batch = Y[:, i:i+batch_size]

            softmax_out, caches = forward_propagation(X_batch, parameters)
            gradients = backpropagation(Y_batch, softmax_out, caches)
            update_parameters(parameters, caches, gradients, learning_rate)
            cost = cross_entropy(softmax_out, Y_batch)
            epoch_cost += np.squeeze(cost)
            
        costs.append(epoch_cost)
        if print_cost and epoch % (epochs // 500) == 0:
            print("Cost after epoch {}: {}".format(epoch, epoch_cost))
        if epoch % (epochs // 100) == 0:
            learning_rate *= 0.98
            
    if plot_cost:
        plt.plot(np.arange(epochs), costs)
        plt.xlabel('# of epochs')
        plt.ylabel('cost')
        plt.show()
    return parameters

window_size = 3
emb_size = 2
text = 'To generate training data, we tokenize text first. There are many techniques out there when it comes to tokenize text data, such as getting rid of words appearing in very high or very low frequency. I just split the text with a simple regex since the focus of the article is not tokenization.'
tokens = tokenize(text)
[id_to_word, word_to_id] = mapping(tokens)
vocab_size = len(id_to_word)
[x, y] = generate_training_data(tokens, word_to_id, window_size)
m = y.shape[1]
y_one_hot = np.zeros((vocab_size, m))
y_one_hot[y.flatten(), np.arange(m)] = 1
softmax_out, caches = forward_propagation(x, initialize_parameters(vocab_size, emb_size))
skipgram_model_training(x, y_one_hot, vocab_size, emb_size, 0.01, 1000)