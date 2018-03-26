from .nn.op import HMMlayer
import pickle
import re, os

class SegModel():
    def __init__(self):
        model = open(os.getcwd()+"/Neu/model/seg.model", 'rb')
        self.word_dict = pickle.load(open("Neu/model/word_dict", 'rb'))
        self.transitionProb = pickle.load(model)
        self.emissionProb = pickle.load(model)
        self.word_list = pickle.load(model)
        self.pi = pickle.load(model)
        self.states = pickle.load(model)

    def catchStr(self, sentence):
        l=[]
        l=re.split('(,|。|、|———|/|●)', sentence)
        values = l[::2]
        delimiters = l[1::2] + ['']

        return [v + d for v, d in zip(values, delimiters)]

    def convertSentence(self, sentence):
        l=[]
        for word in  sentence:
            try:
                l.append(self.word_list.index(word))
            except ValueError:
                l.append(self.word_list.index('。'))

        return l

    def outPutResult(self, sentence, s_seq, term_list, states):
        s = ''
        for i in range(len(sentence)):
            tag = states[s_seq[i]]
            if tag == 'E' or tag == 'S':
                # print(term_list[o_seq[i]], end=' ')
                s += sentence[i] + " "
            else:
                # print(term_list[o_seq[i]], end='')
                s += sentence[i]
        return s

    def mfm(self, string, max_len=5):
        """
        Max forward match
        """

        def getSeg(text):
            if not text:
                return ''
            if len(text) == 1:
                return text
            if text in self.word_dict:
                return text
            else:
                small = len(text) - 1
                text = text[0:small]
                return getSeg(text)

        max_len = max_len
        result_str = ''
        result_len = 0
        unk = ''
        while string:
            tmp_str = string[0:max_len]
            seg_str = getSeg(tmp_str)
            seg_len = len(seg_str)
            if seg_len == 1:
                unk += seg_str
            else:
                if unk != '':
                    s = self.hmm(unk)
                    result_str = result_str + s
                    unk = ''
            result_len = result_len + seg_len

            if seg_str.strip() and seg_len != 1:
                result_str = result_str + seg_str + ' '
            string = string[seg_len:]
        result_str = result_str + ' ' + self.hmm(unk)

        return result_str




    def hmm(self, sentence):  # sentence 为分词后的数组形式

        s=''
        # print(self.word_list)
        windows = self.catchStr(sentence)
        # print(windows)
        for s_window in windows:

            if s_window=='':
                continue
            o_seq = self.convertSentence(s_window+"。")
            s_seq=HMMlayer(o_seq, self.transitionProb, self.emissionProb, self.pi)
            s += self.outPutResult(s_window, s_seq, self.word_list, self.states)
        return s

    def cut(self, string):
        result = self.mfm(string)
        return result