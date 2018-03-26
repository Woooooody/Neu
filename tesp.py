# -*- coding: utf-8 -*-
import Neu
import sys, os
import jieba

sentence = '东北大学原东北工学院，坐落于辽宁省沈  阳市，著名爱国将领张学良曾担任校长，现任校长赵继'
r = Neu.ner(sentence)
for i in r:
    print(i)

s = Neu.cut(sentence)
print(s)
