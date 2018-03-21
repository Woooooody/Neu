from .nn.op import lookup, LSTMlayer, CRFlayer, concat, Linear
from .util import ner_parameters
import numpy as np

class NerModel:
    def __init__(self):
        self.paras = ner_parameters("Neu/model/ner.model")
        self.embedding = self.paras['embeddings']
        self.word2id = self.paras["word2id"]
        self.id2tag = self.paras["id2tag"]
        self.fWi = np.array(self.paras["fWi"]).reshape((100,200))
        self.fbi = np.array(self.paras["fbi"]).reshape((100))
        self.fWf = np.array(self.paras["fWf"]).reshape((100,200))
        self.fbf = np.array(self.paras["fbf"]).reshape((100))
        self.fWu = np.array(self.paras["fWu"]).reshape((100,200))
        self.fbu = np.array(self.paras["fbu"]).reshape((100))
        self.fWo = np.array(self.paras["fWo"]).reshape((100,200))
        self.fbo = np.array(self.paras["fbo"]).reshape((100))
        self.bWi = np.array(self.paras["bWi"]).reshape((100,200))
        self.bbi = np.array(self.paras["bbi"]).reshape((100))
        self.bWf = np.array(self.paras["bWf"]).reshape((100,200))
        self.bbf = np.array(self.paras["bbf"]).reshape((100))
        self.bWu = np.array(self.paras["bWu"]).reshape((100,200))
        self.bbu = np.array(self.paras["bbu"]).reshape((100))
        self.bWo = np.array(self.paras["bWo"]).reshape((100,200))
        self.bbo = np.array(self.paras["bbo"]).reshape((100))
        self.W_tag = np.array(self.paras["W_tag"]).reshape((200, 7))
        self.b_tag = np.array(self.paras["b_tag"]).reshape((7))
        self.trans = np.array(self.paras["transitions"])

    def ner(self, word_list):
        ids = [self.word2id[word] for word in word_list]

        emb = lookup(self.embedding, ids)   # (n, 100)
        forward = LSTMlayer(emb, self.fWi, self.fbi, self.fWf, self.fbf, self.fWu, self.fbu, self.fWo, self.fbo)
        # print(forward)

        emb.reverse()
        backward = LSTMlayer(emb, self.bWi, self.bbi, self.bWf, self.bbf, self.bWu, self.bbu, self.bWo, self.bbo)
        backward = np.flip(backward, 0)
        top_recur = concat(forward, backward, axis=1)
        tag_score = Linear(top_recur, self.W_tag, self.b_tag)
        # print(tag_score)
        tag_ids = CRFlayer(tag_score, self.trans)
        tag_list = [self.id2tag[id] for id in tag_ids]
        # print(tag_list)

        return tag_list
