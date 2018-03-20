
from .nn.op import lookup, LSTMlayer, CRFlayer
from .util import ner_parameters
import numpy as np


class NerModel:
    def __init__(self):
        self.paras = ner_parameters("path")
        self.embedding = self.paras['embeding']
        self.word2id = self.paras["word2id"]
        self.id2tag = self.paras["id2tag"]
        self.Wi = np.array(self.paras["Wi"])
        self.bi = np.array(self.paras["bi"])
        self.Wf = np.array(self.paras["Wf"])
        self.bf = np.array(self.paras["bf"])
        self.Wu = np.array(self.paras["Wu"])
        self.bu = np.array(self.paras["bu"])
        self.Wo = np.array(self.paras["Wo"])
        self.bo = np.array(self.paras["bo"])
        self.W_tag = np.array(self.paras["W_tag"])
        self.trans = np.array(self.paras["transitions"])

    def ner(self, word_list):
        ids = [self.word2id[word] for word in word_list]
        emb = lookup(self.embedding, ids)   # (n, 100)
        foward = LSTMlayer(emb, self.Wi, self.bi, self.Wf, self.bf, self.Wu, self.bu, self.Wo, self.bo)
        backward = LSTMlayer(emb.reverse(), self.Wi, self.bi, self.Wf, self.bf, self.Wu, self.bu, self.Wo, self.bo)
        backward.reverse()
        top_recur = foward.extend(backward)
        tag_score = np.dot(np.array(top_recur), self.W_tag)
        # tag_score = np.array([[0.8,0.1,0.1],[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8]])
        # self.trans = np.array([[0.4,0.5,0.1],[0.3,0.4,0.3],[0.5,0.1,0.4]])
        tag_ids = CRFlayer(tag_score, self.trans)
        tag_list = [self.id2tag[id] for id in tag_ids]

        return tag_list
