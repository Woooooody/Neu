import numpy as np


def lookup(vocab, ids):
    emb = []
    for i in ids:
        emb.append(vocab[i])
    return emb

class LSTMparam:
    def __init__(self, wg, wi, wf, wo, bg, bi, bf, bo):
        # weight matrices
        self.wg = wg
        self.wi = wi
        self.wf = wf
        self.wo = wo
        # bias terms
        self.bg = bg
        self.bi = bi
        self.bf = bf
        self.bo = bo


class LSTMstate:
    def __init__(self, hidden_size):
        self.g = np.zeros(hidden_size)
        self.i = np.zeros(hidden_size)
        self.f = np.zeros(hidden_size)
        self.o = np.zeros(hidden_size)
        self.s = np.zeros(hidden_size)
        self.h = np.zeros(hidden_size)



def LSTMlayer(emb, Wi, bi, Wf, bf, Wg, bg, Wo, bo):
    # lstm_para = LSTMparam(Wg, Wi, Wf, Wo, bg, bi, bf, bo)
    # lstm_state = LSTMstate(100)
    # cell = LSTMcell(lstm_para, lstm_state)
    #
    # out = []
    # for xi in emb:
    #     h_pre = lstm_state.h
    #     s_pre = lstm_state.s
    #     cell.update_cell(xi, h_pre, s_pre)
    #     out.append(lstm_state.h)
    # return np.array(out)


    h_pre = np.zeros((100))
    cell_state = np.zeros(bi.shape)
    out = []
    for xi in emb:
        # print(h_pre)
        xi = np.array(xi)
        input = sigmoid(np.dot(Wi, concat(xi, h_pre)) + bi)
        # print(input.shape)
        forget = sigmoid(np.dot(Wf, concat(xi, h_pre)) + bf)
        cell_update = tanh(np.dot(Wg, concat(xi, h_pre)) + bg)
        cell_state = (forget * cell_state) + (input * cell_update)
        output = sigmoid(np.dot(Wo, concat(xi, h_pre)) + bo)
        h_out = (tanh(output) * cell_state)
        # print(tanh(cell_state))
        out.append(h_out)
        h_pre = h_out
        # print(h_out)
    return np.array(out)


class LSTMcell:
    def __init__(self, lstm_param, lstm_state):
        # store reference to parameters and to activations
        self.state = lstm_state
        self.param = lstm_param
        # non-recurrent input concatenated with recurrent input
        self.xc = None

    def update_cell(self, x, s_prev=None, h_prev=None):
        # if this is the first lstm node in the network
        if s_prev is None: s_prev = np.zeros_like(self.state.s)
        if h_prev is None: h_prev = np.zeros_like(self.state.h)
        # save data for use in backprop
        self.s_prev = s_prev
        self.h_prev = h_prev

        # concatenate x(t) and h(t-1)
        xc = np.hstack((x, h_prev))
        self.state.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg)
        self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi)
        self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)
        self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f
        self.state.h = self.state.s * self.state.o

        self.xc = xc

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