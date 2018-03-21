import codecs
def ner_parameters(path):
    paras = {}
    keys = ['embeddings', 'fWi', 'fWu', 'fWf', 'fWo', 'fbi', 'fbu', 'fbf', 'fbo', 'bWi', 'bWu', 'bWf', 'bWo', 'bbi',
            'bbu', 'bbf', 'bbo', 'W_tag', 'b_tag', 'transitions', 'word2id', 'id2word']
    k = 0
    paras[keys[k]] = []
    file = codecs.open(path, 'r', encoding='utf-8')
    count = 0
    for line in file:
        if line.strip() == '#':
            k += 1
            paras[keys[k]] = []
            continue
        if line.strip() == '?':
            k += 1
            paras[keys[k]] = {}
            paras[keys[k + 1]] = {}
            continue
        tmp = line.split()
        if len(tmp) == 1:
            w2i = paras[keys[k]]
            i2w = paras[keys[k + 1]]
            w2i[tmp[0]] = count
            i2w[count] = tmp[0]
            count += 1
        else:
            tmp = [float(i) for i in tmp]
            paras[keys[k]].append(tmp)
    paras['id2tag'] = {0:'I-LOC', 1:'B-ORG', 2:'I-PER', 3:'O', 4:'I-ORG', 5:'B-LOC', 6:'B-PER'}
    file.close()
    return paras

def parser_parameters():
    pass

if __name__ == '__main__':
    paras = ner_parameters('ner.model')
    embeddings = paras['embeddings']
    word2id = paras['word2id']
    print(len(embeddings))
