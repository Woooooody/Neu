import numpy as np


def lookup(word_list, vocab, word2id):
    # ids = [word2id[word] if word in word2id else word2id['UNK'] for word in word_list]
    ids = [word2id[word] for word in word_list]
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

    h_pre = np.zeros((100))
    cell_state = np.zeros(bi.shape)
    out = []
    for xi in emb:
        xi = np.array(xi)
        input = sigmoid(np.dot(Wi, concat(xi, h_pre)) + bi)
        forget = sigmoid(np.dot(Wf, concat(xi, h_pre)) + bf)
        cell_update = tanh(np.dot(Wg, concat(xi, h_pre)) + bg)
        cell_state = (forget * cell_state) + (input * cell_update)
        output = sigmoid(np.dot(Wo, concat(xi, h_pre)) + bo)
        h_out = (tanh(output) * cell_state)
        out.append(h_out)
        h_pre = h_out
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
        self.state.g = np.tanh(np.dot(self.param.wg) + self.param.bg)
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

def HMMlayer(o_sequence, A, B, pi):
    '''
        维特比算法：
        input:
            o_sequence:观测序列
            :条件转移概率
            :观测概率
            pi:初始状态概率

        return 符合最优的状态序列
    '''
    len_status = len(pi)

    status_record = {i: [[0, 0] for j in range(len(o_sequence))] for i in range(len_status)}

    for i in range(len(pi)):
        status_record[i][0][0] = pi[i] * B[i, o_sequence[0]]  # t时刻下状态为i的所有单个路径的最大概率,δt(i)=πi * bi(o1)
        status_record[i][0][1] = 0  # t时刻下状态为i的最大概率路径的第t-1个节点,Ψt(i)=0

    for t in range(1, len(o_sequence)):
        for i in range(len_status):
            max = [-1, 0]
            for j in range(len_status):  # max(δt(j)*aji) i = 1,2,...,N
                tmp_prob = status_record[j][t - 1][0] * A[j, i]  # δt(j) * A[j][i]
                if tmp_prob > max[0]:
                    max[0] = tmp_prob
                    max[1] = j

            status_record[i][t][0] = max[0] * B[i, o_sequence[t]]
            status_record[i][t][1] = max[1]

    max = 0
    max_idx = 0
    t = len(o_sequence) - 1
    for i in range(len_status):
        if max < status_record[i][t][0]:
            max = status_record[i][t][0]
            max_idx = i

    state_sequence = []  # 栈结构
    state_sequence.append(max_idx)
    while (t > 0):
        max_idx = status_record[max_idx][t][1]
        state_sequence.append(max_idx)
        t -= 1
    state_sequence.reverse()
    return state_sequence



def concat(tensor1, tensor2, axis=0):
    return np.concatenate((tensor1, tensor2), axis=axis)


def tanh(tensor):
    return np.tanh(tensor)


def sigmoid(tensor):
    for i in range(len(tensor)):
        tensor[i] = 1 / (1 + np.exp(-tensor[i]))
    return tensor
