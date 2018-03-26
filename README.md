===
Neu
===
中文自然语言处理工具包--东北大学自然语言处理实验室(http://www.nlplab.com)

Neu是东北大学自然语言处理实验室在2014开发的NiuParser的Python版本.

NiuParser(http://www.niuparser.com), NiuParser支持所有的中文自然语言处理底层技术.

Neu初步预计有四个模块,分别是分词,词性标注,实体识别,句法分析.

安装方便,可直接pip安装,不依赖于繁多的包,敬请期待.!

有任何问题可联系wujinhang0729@163.com, huxiao1318@163.com


Installation
============

Using ``pip``::

    pip3 install Neu (目前只支持Python3)

Using ``git``::

    clone https://github.com/Woooooody/Neu (解压后放在python的site package目录)

Useage
======

![help](https://github.com/Woooooody/Neu/blob/master/images/help.png)

License
=======

*************************************************************************************
*  					 			    Neu SDK   					                    *
*					                                                                *
* 				Copyright (c) 2017-2018 NiuParser All rights reserved.              *
*                                                                                   *
*  				                 www.nlplab.com                                     *
*************************************************************************************

Authors and Contributors
========================

Original authors are Wu Jinhang <wujinhang0729@gmail.com> and Huxiao <https://github.com/huxiao>. 

Method
=======

For word segment we use the maximum forward match algorithm and HMM(Hidden Markov Model)
Using viterbi algorithm to caculate the maximum rotate.

For sequence label model, we use the character level inputs for  LSTM(Long Short Time Memory) and CRF(Conditional Random Field) model .
