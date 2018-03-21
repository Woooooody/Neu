import numpy as np


def lookup(vocab, ids):
    emb = []
    for i in ids:
        emb.append(vocab[i])
    return emb


def LSTMlayer(emb, Wi, bi, Wf, bf, Wu, bu, Wo, bo):
    h_pre = np.zeros((100))
    cell_state = np.zeros(bi.shape)
    out = []
    for xi in emb:
        # print(h_pre)
        xi = np.array(xi)
        input = sigmoid(np.dot(Wi, concat(h_pre, xi)) + bi)
        # print(input.shape)
        forget = sigmoid(np.dot(Wf, concat(h_pre, xi)) + bf)
        cell_update = tanh(np.dot(Wu, concat(h_pre, xi)) + bu)
        cell_state = (forget * cell_state) + (input * cell_update)
        output = sigmoid(np.dot(Wo, concat(h_pre, xi)) + bo)
        h_out = (output * tanh(cell_state))
        # print(tanh(cell_state))
        out.append(h_out)
        h_pre = h_out
        # print(h_out)
    return np.array(out)

def Linear(input, W, b):
    return np.dot(input, W) + b

def CRFlayer(B, A):
    """
    Using viterbi algorithm to decoder
    :param B: LSTM prediction output
    :param A: The transition probability matrix of tags
    :return: the final score through CRF layer of each tag of each word
    """
    delta = np.zeros(B.shape)
    psi = np.zeros(B.shape)     # 记录路径
    for i in range(B.shape[1]):
        delta[0][i] = B[0][i]

    for t in range(1, B.shape[0]):
        for i in range(B.shape[1]):
            max = 0.0
            index = 0
            for j in range(B.shape[1]):
                if max < delta[t-1][j] + A[j][i] + B[t][i]:
                    max = delta[t-1][j] + A[j][i] + B[t][i]
                    index = j
            delta[t][i] = max
            psi[t][i] = index

    #   go back to find the maximum rotate
    max = 0
    max_id = 0
    t = len(delta) - 1
    for i in range(B.shape[1]):
        if delta[-1][i] > max:
            max = delta[-1][i]
            max_id = i
    state = [max_id]
    while t > 0:
        max_id = int(psi[t][max_id])
        state.append(max_id)
        t -= 1
    state.reverse()
    return state

def concat(tensor1, tensor2, axis=0):
    return np.concatenate((tensor1, tensor2), axis=axis)


def tanh(tensor):
    return np.tanh(tensor)


def sigmoid(tensor):
    for i in range(len(tensor)):
        tensor[i] = 1 / (1 + np.exp(-tensor[i]))
    return tensor

if __name__ == "__main__":
    h_pre = Tensor.zero([1, 5])
    input = sigmoid(Wi * concat(h_pre, xi) + bi)