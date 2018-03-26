# -*- coding: utf-8 -*-
"""
*************************************************************************************
*  					 			    Neu SDK   					                    *
*					                                                                *
* 				Copyright (c) 2017-2018 NiuParser All rights reserved.              *
*                                                                                   *
*  				                 www.nlplab.com                                     *
*************************************************************************************
Project Name : Neu V1.0
Author       : Wu Jinhang, Hu Xiao
Email        : wujinhang0729@gmail.com
Create Time  : 2018/03/26/19:31:05
Copyright    : Copyright (c) 2018 NEU NLP LAB. All rights reserved.
"""
from .SegModel import SegModel
from .PosModel import PosModel
from .NerModel import NerModel
from .ParserModel import ParserModel
import os

dir_path = os.path.dirname(os.path.abspath(__file__))

cut = SegModel(dir_path+"/model/seg.model").cut
# pos = NerModel(dir_path+"/model/pos.model").ner
ner = NerModel(dir_path+"/model/ner.model").ner
# parser = ParserModel().parser


def help():
    print("\033[0;32m%s\033[0m" % "******************************************************************")
    print("\033[0;32m%s\033[0m" % "Project Name : Neu V1.0")
    print("\033[0;32m%s\033[0m" % "Author       : Wu Jinhang, Hu Xia")
    print("\033[0;32m%s\033[0m" % "Email        : wujinhang0729@gmail.com")
    print("\033[0;32m%s\033[0m" % "Create Time  : 2018/03/26/19:31:05")
    print("\033[0;32m%s\033[0m" % "Copyright    : Copyright (c) 2018 NEU NLP LAB. All rights reserved.")
    print("\033[0;32m%s\033[0m" % "******************************************************************\n")

    print("\033[1;31m%s\033[0m" % "Neu.cut() 分词 : \n")
    print("\033[0;30m%s\033[0m" % "input is a string of chinese sentence\n")
    print("\033[0;30m%s\033[0m" % "output is a string of the result of word segmentation\n")
    print("\033[0;30m%s\033[0m" % "--------------------- Example -----------------------\n")
    print("\033[0;30m%s\033[0m" % "import Neu\n")
    print("\033[0;30m%s\033[0m" % "result = Neu.cut('东北大学坐落于辽宁沈阳,原名东北工学院')\n")
    print("\033[0;30m%s\033[0m" % "print(result)\n")
    print("\033[0;30m%s\033[0m" % "output: 东北大学 坐落 于 辽宁 沈阳 , 原名 东北 工学院\n\n")

    print("\033[1;31m%s\033[0m" % "Neu.ner() 实体识别 : \n")
    print("\033[0;30m%s\033[0m" % "input is a string of chinese sentence\n")
    print("\033[0;30m%s\033[0m" % "output is a list of the result of ner\n")
    print("\033[0;30m%s\033[0m" % "--------------------- Example -----------------------\n")
    print("\033[0;30m%s\033[0m" % "import Neu\n")
    print("\033[0;30m%s\033[0m" % "result = Neu.ner('东北大学坐落于辽宁沈阳')\n")
    print("\033[0;30m%s\033[0m" % "print(result)\n")
    print("\033[0;30m%s\033[0m" % "output: ['东 B-ORG', '北 I-ORG', '大 I-ORG', '学 I-ORG', '坐 O', '落 O', '于 O', '辽 B-LOC', '宁 I-LOC', '沈 B-LOC', '阳 I-LOC']\n\n")




