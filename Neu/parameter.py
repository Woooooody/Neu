class Parameter():
    def __init__(self, dir):
        self.embedding = []
        self.word2id = []
        self.id2tag = []

    def load_vocab(self, dir):
        f = open(dir, 'r')
        for line in f:
            self.embedding.append(line)