from .util import parser_parameters
import numpy as np


class ParserModel:
    def parser(self):
        def __init__(self):
            self.paras = parser_parameters("path")
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

        def parser():
            pass
