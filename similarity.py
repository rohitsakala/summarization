#!/usr/bin/env python
# coding: utf-8

import gensim
import sklearn
import os
import re
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk.tokenize import sent_tokenize,word_tokenize,RegexpTokenizer
import nltk
from functools import reduce
import numpy as np


word2vecModel = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)


def getsimilarCount(sent_1, sent_2):
    vector1 = reduce(np.add,[word2vecModel.get_vector(x) if x in word2vecModel.vocab else np.zeros(300) for x in sent_1.strip().split()])
    vector2 = reduce(np.add,[word2vecModel.get_vector(x) if x in word2vecModel.vocab else np.zeros(300) for x in sent_2.strip().split()])
    return word2vecModel.cosine_similarities(vector1, [vector2])