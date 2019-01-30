#/usr/bin/env python

# -*- coding: utf-8 -*-
from __future__ import division, unicode_literals
import os, sys
import networkx as nx
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
import mongoOps
from matplotlib import pyplot as plt
import json
import pandas as pd
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import itertools
from operator import itemgetter
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from nltk.tokenize import sent_tokenize,word_tokenize,RegexpTokenizer
from summa.summarizer import summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer 
from sumy.summarizers.lex_rank import LexRankSummarizer
from summa.keywords import keywords
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
from gensim import corpora
import gensim
import re
from nltk import pos_tag, ne_chunk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import sent_tokenize
import nltk
from rouge import Rouge
from pythonrouge.pythonrouge import Pythonrouge
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from similarity import *


verbs_pos = ["VB","VBD","VBG","VBN","VBP","VBZ"]
adverb_pos = ["RB", "RBR","RBS"]
nouns_pos = ["NN", "NNS", "NNP" , "NNPS", "GP" ]
adjective_pos = ["JJ","JJR","JJS"]

# Debate Id
debatesList = ["5a42549689d32a43eb059cff","5a42549689d32a43eb059cff_1","5a42537189d32a43eb04e6c7","5a42653e89d32a43eb1034cf","5a4257a889d32a43eb07987b","5a426fb289d32a43eb15d389","5a426fb289d32a43eb15d389_1"]
#debatesList = ["5a426fb289d32a43eb15d389"]

scores = {}
scores['textRank'] = {}
scores['textRank']['rouge-1'] = {}
scores['textRank']['rouge-2'] = {}
scores['textRank']['rouge-l'] = {}
scores['textRank']['rouge-1']['p'] = 0
scores['textRank']['rouge-1']['r'] = 0
scores['textRank']['rouge-1']['f'] = 0
scores['textRank']['rouge-2']['p'] = 0
scores['textRank']['rouge-2']['r'] = 0
scores['textRank']['rouge-2']['f'] = 0
scores['textRank']['rouge-l']['p'] = 0
scores['textRank']['rouge-l']['r'] = 0
scores['textRank']['rouge-l']['f'] = 0
scores['nenkova'] = {}
scores['nenkova']['rouge-1'] = {}
scores['nenkova']['rouge-2'] = {}
scores['nenkova']['rouge-l'] = {}
scores['nenkova']['rouge-1']['p'] = 0
scores['nenkova']['rouge-1']['r'] = 0
scores['nenkova']['rouge-1']['f'] = 0
scores['nenkova']['rouge-2']['p'] = 0
scores['nenkova']['rouge-2']['r'] = 0
scores['nenkova']['rouge-2']['f'] = 0
scores['nenkova']['rouge-l']['p'] = 0
scores['nenkova']['rouge-l']['r'] = 0
scores['nenkova']['rouge-l']['f'] = 0
scores['lexRank'] = {}
scores['lexRank']['rouge-1'] = {}
scores['lexRank']['rouge-2'] = {}
scores['lexRank']['rouge-l'] = {}
scores['lexRank']['rouge-1']['p'] = 0
scores['lexRank']['rouge-1']['r'] = 0
scores['lexRank']['rouge-1']['f'] = 0
scores['lexRank']['rouge-2']['p'] = 0
scores['lexRank']['rouge-2']['r'] = 0
scores['lexRank']['rouge-2']['f'] = 0
scores['lexRank']['rouge-l']['p'] = 0
scores['lexRank']['rouge-l']['r'] = 0
scores['lexRank']['rouge-l']['f'] = 0
scores['own'] = {}
scores['own']['rouge-1'] = {}
scores['own']['rouge-2'] = {}
scores['own']['rouge-l'] = {}
scores['own']['rouge-1']['p'] = 0
scores['own']['rouge-1']['r'] = 0
scores['own']['rouge-1']['f'] = 0
scores['own']['rouge-2']['p'] = 0
scores['own']['rouge-2']['r'] = 0
scores['own']['rouge-2']['f'] = 0
scores['own']['rouge-l']['p'] = 0
scores['own']['rouge-l']['r'] = 0
scores['own']['rouge-l']['f'] = 0

#debateGlobalId = "5a42549689d32a43eb059cff" #- GST
#debateGlobalId = "5a42537189d32a43eb04e6c7" # - SBI

#Gold Summary Folder
goldSummary_Path = "./goldSummary/"

def my_key(dict_key):
	try:
		return int(dict_key)
	except ValueError:
		return dict_key

def debateFind(debateId):
	docsList = []
	data = mongoOps.getDocument("synopsis","debates",debateId)
	try:
		data = data["mattersMap"]
	except:
		pass
	keys = sorted(data)
	for key in keys:
		if key != "_id" and key != "keywords" and key != "summary":
				docsList.append(data[str(key)]['speech'])
	return docsList

def topicFind(speechsList,topicModel,leng):
	if topicModel == "NMF":
		return NMF_run(speechsList,leng)
	if topicModel == "LDA":
		return LDA_run(speechsList,leng)
	if topicModel == "TextRank":
		return TextRank_run(speechsList,leng)
	if topicModel == "LSA":
		return LSA_run(speechsList,leng)

def NMF_run(docsList,leng):
	documents = docsList
	no_features = 10000

	# NMF is able to use tf-idf
	tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
	tfidf = tfidf_vectorizer.fit_transform(documents)
	tfidf_feature_names = tfidf_vectorizer.get_feature_names()

	if int(leng) == 500:
		no_topics = 3
	if int(leng) == 1000:
		no_topics = 15
	if int(leng) == 1500:
		no_topics = 15


	#Run NMF
	nmf = NMF(n_components=no_topics, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

	if int(leng) == 500:
		no_top_words = 3
	if int(leng) == 1000:
		no_top_words = 15
	if int(leng) == 1500:
		no_top_words = 15

	return display_topics(nmf, tfidf_feature_names, no_top_words)

def LDA_run(docsList,leng):
	documents = docsList
	no_features = 10000

	# NMF is able to use tf-idf
	tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
	tf = tf_vectorizer.fit_transform(documents)
	tf_feature_names = tf_vectorizer.get_feature_names()

	if int(leng) == 500:
		no_topics = 2
	if int(leng) == 1000:
		no_topics = 10
	if int(leng) == 1500:
		no_topics = 15


	#Run NMF
	lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

	if int(leng) == 500:
		no_top_words = 2
	if int(leng) == 1000:
		no_top_words = 10
	if int(leng) == 1500:
		no_top_words = 15

	return display_topics(lda, tf_feature_names, no_top_words)

def display_topics(model, feature_names, no_top_words):
	s = set()
	for topic_idx, topic in enumerate(model.components_):
		#print "Topic %d:" % (topic_idx)
		featureList = [feature_names[i]
								for i in topic.argsort()[:-no_top_words - 1:-1]]
		#print featureList
		for feat in featureList:
			s.add(feat)
	return s

def TextRank_run(speechsList,leng):
	s = set()
	document = ""
	for speech in speechsList:
		document = document + speech + " "
	if int(leng) == 500:
		for keyword in keywords(document,words=4):
			s.add(keyword)
	if int(leng) == 1000:
		for keyword in keywords(document,words=10):
			s.add(keyword)
	if int(leng) == 1500:
		for keyword in keywords(document,words=10):
			s.add(keyword)
	return s

def LSA_run(speechsList,leng):
	vectorizer = TfidfVectorizer(stop_words='english', max_features= 1000, max_df = 0.5, smooth_idf=True)
	X = vectorizer.fit_transform(speechsList)

	if int(leng) == 500:
		n_components = 4
	if int(leng) == 1000:
		n_components = 10
	if int(leng) == 1500:
		n_components = 15

	svd_model = TruncatedSVD(n_components, algorithm='randomized', n_iter=100, random_state=122)
	svd_model.fit(X)
	terms = vectorizer.get_feature_names()

	if int(leng) == 500:
		no_top_words = 4
	if int(leng) == 1000:
		no_top_words = 10
	if int(leng) == 1500:
		no_top_words = 15

	s = set()

	for i, comp in enumerate(svd_model.components_):
		terms_comp = zip(terms, comp)
		sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:no_top_words]
		for t in sorted_terms:
			s.add(t[0])
	return s

def removeSentSplitProbs(data): # TODO Abbreivaition sentence split
	data = re.sub(r'\s+', ' ', data)
	if "Rs. " in data:
		data = data.replace('Rs. ','Rs.')
	if "hon. " in data:
		data = data.replace('hon. ','hon.')
	if "Hon. " in data:
		data = data.replace('Hon. ','Hon.')
	if "No. " in data:
		data = data.replace('No. ','No.')
	if "Cr. " in data:
		data = data.replace('Cr. ','Cr.')
	return data

def sentenceFind(speechsList):
	sentenceList = []
	for speech in speechsList:
		speech = removeSentSplitProbs(speech)
		punkt_param = PunktParameters()
		abbreviation = ['hon.','rs','no']
		punkt_param.abbrev_types = set(abbreviation)
		tokenizer = PunktSentenceTokenizer(punkt_param)
		tokenizer.tokenize(speech)
		sentencesTemp = sent_tokenize(speech)
		for sentence in sentencesTemp:
			sentenceList.append(sentence)
	return sentenceList

def stemStopWordPOSSpeeches(speechsList):
	stop_words = set(stopwords.words('english'))
	newSpeechList = []
	ps = PorterStemmer()
	newspeech = ""
	for speech in speechsList:
		newspeech = ""
		words = word_tokenize(speech)
		newwords = []
		for word in words:
			if not word in stop_words:
				newwords.append(ps.stem(word))
				#newwords.append(word)
		words = newwords
		words = pos_tag(words)
		org_words = ne_chunk(words)
		for tagged in org_words:
			if isinstance(tagged,tuple):
				if tagged[1] in adverb_pos or tagged[1] in adjective_pos or tagged[1] in verbs_pos:
					pass
				else:
					newspeech = newspeech + " " + tagged[0]
			else:
				newspeech = newspeech + " " + tagged.leaves()[0][0]
		newSpeechList.append(newspeech)
	return newSpeechList

def processSentence(sentenceList):
	proSentenceList = list()
	for sent in sentenceList:
		sent = re.sub(r'\W+', ' ', sent.lower())
		proSentenceList.append(sent)
	return proSentenceList

def stemSentence(sentence):
    st = PorterStemmer()
    words = [st.stem(word.lower()) for word in re.sub("[\.\,\!\?;\:\(\)\[\]\'\"]$", '', sentence.rstrip()).split()]
    return words 

def topSentenceFind(topicList,sentenceList):
	topSentenceList = set()
	for topic in topicList:
		for sentence in sentenceList[:]:
			if topic in re.sub(r'\W+', ' ', sentence.lower()):
				topSentenceList.add(sentence)
				sentenceList.remove(sentence)
	return list(topSentenceList)

def getWordCount(speechsList):
	if len(speechsList) == 0:
		return 0
	count = 0
	if isinstance(speechsList,set) or isinstance(speechsList,list):
		for speech in speechsList:
			wordList = word_tokenize(speech)
			count = count + len(wordList)
	else:
		wordList = word_tokenize(speechsList)
		count = count + len(wordList)
	return count

def removeDuplicateSent(sentenceList):
	sentenceList = list(sentenceList)
	for sentence_1 in sentenceList[:]:
		for sentence_2 in sentenceList[:]:
			if sentence_1 != sentence_2:
				if float(getsimilarCount(sentence_1,sentence_2)) > 0.8:
					#print("++++++++++++++++++++")
					#print(sentence_1)
					#print(sentence_2)
					#print("++++++++++++++++++++")
					if len(sentence_1.split()) > len(sentence_2.split()): # Choosing based on length
						try:
							sentenceList.remove(sentence_2)
						except:
							pass
					else:
						try:
							sentenceList.remove(sentence_1)
						except:
							pass
	#print(len(sentenceList))
	return sentenceList


'''def removeDuplicateSent(topSentenceList):
	topSentenceList = list(topSentenceList)
	tfidf_vectorizer = TfidfVectorizer()
	tfidf_matrix = tfidf_vectorizer.fit_transform(topSentenceList)
	duplicatedSentenceList = set()
	count = 0
	for sentence in range(len(topSentenceList)):
		value = cosine_similarity(tfidf_matrix[sentence], tfidf_matrix)
		maxLengthIndex = sentence
		for index, val in enumerate(value[0]):
			if val > 0.55 and index != sentence:  # Perfect for GST TODO check for other bills too
				#print topSentenceList[sentence]
				#print topSentenceList[index]
				#print "+++++++++++++++++++++++++++++++"
				if len(topSentenceList[maxLengthIndex]) < len(topSentenceList[index]): # Selecting the larger sentence of the two similar sentences
					maxLengthIndex = index
		duplicatedSentenceList.add(topSentenceList[maxLengthIndex])
	return duplicatedSentenceList

def removeDuplicateSent(sentenceList): # https://nlpforhackers.io/wordnet-sentence-similarity/
	dupTopSentenceList = sentenceList
	for sentence_1 in sentenceList:
		for sentence_2 in sentenceList:
			if sentence_similarity(sentence_1,sentence_2) > 0.5:
				if len(sentence_1.split()) > len(sentence_2.split()): # Choosing based on length
					dupTopSentenceList.remove(sentence_2)
				else:
					dupTopSentenceList.remove(sentence_1)
	return dupTopSentenceList'''

def printSpeeches(speechsList):
	for speech in speechsList:
		print(speech)
		print("+++++++++++++")

def sentSimilarityWordWise(dupTopSentenceList,line,topicList):
	line = re.sub(r'\W+', ' ', line.lower())
	uniqueLine = set()
	for word in line.split():
		uniqueLine.add(word)
	lenLine = len(uniqueLine)
	d = {}
	for word in line.split():
		d[word] = True
	similarCheck = False
	for sent in dupTopSentenceList:
  		sent = re.sub(r'\W+', ' ', sent.lower())
  		#print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
  		#print sent
  		uniqueSent = set()
  		for word in sent.split():
  			uniqueSent.add(word)
  		lenSent = len(uniqueSent)
  		count = set()
  		for word in sent.split():
  			try:
  				if d[word]:
  					count.add(word)
  			except:
  				pass
  			minLen = 0
  			if lenLine > lenSent:
  				minLen = lenLine
  			else:
  				minLen = lenSent
  			if len(count) > minLen - 1:
  				#print "True"
  				#a = raw_input()
  				similarCheck = True
  				break
  		if similarCheck == True:
  			break 
	if similarCheck:
		#print line
		#print "True"
		#raw_input()
		return True
	else:
		lineList = list()
		lineList.append(line)
		if topSentenceFind(topicList,lineList):
			#print line
			#print "True-Sub"
			#raw_input()
			return True
		#print line
		#print "False"
		#raw_input()
		return False
	#print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

def claimPremiseSupportSentenceFind(topSentenceList,topicList,annotatedGraph,leng):
	claimPremiseSupportTopSentenceList = set()
	G = annotatedGraph
	if int(leng) == 500:
		leng = 550
	if int(leng) == 1000:
		leng = 1100
	if int(leng) == 1500:
		leng = 1650
	for node,values in G.nodes(data =True):
		if values['entity'] == "Claim":
			for node1,node2 in G.edges(node):
				if G[node1][node2]['relation'] == "supports":
					if sentSimilarityWordWise(topSentenceList,G.nodes[node1]['sent'],topicList):
						claimPremiseSupportTopSentenceList.add(G.nodes[node1]['sent'])
					if sentSimilarityWordWise(topSentenceList,G.nodes[node2]['sent'],topicList):
						claimPremiseSupportTopSentenceList.add(G.nodes[node2]['sent'])
					if getWordCount(claimPremiseSupportTopSentenceList) > leng:
						return claimPremiseSupportTopSentenceList
	return claimPremiseSupportTopSentenceList

def claimOnePremiseSentenceFind(topSentenceList,topicList,annotatedGraph,leng):
	claimOnePremiseTopSentenceList = set()
	G = annotatedGraph
	if int(leng) == 500:
		leng = 550
	if int(leng) == 1000:
		leng = 1100
	if int(leng) == 1500:
		leng = 1650
	for node,values in G.nodes(data =True):
		if values['entity'] == "Claim":
			count = 0
			for node1,node2 in G.edges(node):
				count = count + 1
				if sentSimilarityWordWise(topSentenceList,G.nodes[node1]['sent'],topicList):
					claimOnePremiseTopSentenceList.add(G.nodes[node1]['sent'])
				if sentSimilarityWordWise(topSentenceList,G.nodes[node2]['sent'],topicList):
					claimOnePremiseTopSentenceList.add(G.nodes[node2]['sent'])
				if getWordCount(claimOnePremiseTopSentenceList) > leng:
					return claimOnePremiseTopSentenceList
				if count == 1:
					break
	return claimOnePremiseTopSentenceList

def claimTwoPremiseSentenceFind(topSentenceList,topicList,annotatedGraph,leng):
	claimTwoPremiseTopSentenceList = set()
	G = annotatedGraph
	if int(leng) == 500:
		leng = 550
	if int(leng) == 1000:
		leng = 1100
	if int(leng) == 1500:
		leng = 1650
	for node,values in G.nodes(data =True):
		if values['entity'] == "Claim":
			count = 0
			for node1,node2 in G.edges(node):
				count = count + 1
				if sentSimilarityWordWise(topSentenceList,G.nodes[node1]['sent'],topicList):
					claimTwoPremiseTopSentenceList.add(G.nodes[node1]['sent'])
				if sentSimilarityWordWise(topSentenceList,G.nodes[node2]['sent'],topicList):
					claimTwoPremiseTopSentenceList.add(G.nodes[node2]['sent'])
				if getWordCount(claimTwoPremiseTopSentenceList) > leng:
						return claimTwoPremiseTopSentenceList
				if count == 2:
					break
	return claimTwoPremiseTopSentenceList

def claimThreePremiseSentenceFind(topSentenceList,topicList,annotatedGraph,leng):
	claimThreePremiseTopSentenceList = set()
	G = annotatedGraph
	if int(leng) == 500:
		leng = 550
	if int(leng) == 1000:
		leng = 1100
	if int(leng) == 1500:
		leng = 1650
	for node,values in G.nodes(data =True):
		if values['entity'] == "Claim":
			count = 0
			for node1,node2 in G.edges(node):
				count = count + 1
				if sentSimilarityWordWise(topSentenceList,G.nodes[node1]['sent'],topicList):
					claimThreePremiseTopSentenceList.add(G.nodes[node1]['sent'])
				if sentSimilarityWordWise(topSentenceList,G.nodes[node2]['sent'],topicList):
					claimThreePremiseTopSentenceList.add(G.nodes[node2]['sent'])
				if getWordCount(claimThreePremiseTopSentenceList) > leng:
						return claimThreePremiseTopSentenceList
				if count == 3:
					break
	return claimThreePremiseTopSentenceList

def descConnGraphSentenceFind(topSentenceList,topicList,annotatedGraph,leng):
	descConnGraphTopSentenceList = set()
	G = annotatedGraph
	if int(leng) == 500:
		leng = 550
	if int(leng) == 1000:
		leng = 1100
	if int(leng) == 1500:
		leng = 1650
	for c in sorted(nx.connected_components(annotatedGraph), key=len, reverse=True):
		for s in c:
			if sentSimilarityWordWise(topSentenceList,G.nodes[s]['sent'],topicList):
				descConnGraphTopSentenceList.add(G.nodes[s]['sent'])
			if getWordCount(descConnGraphTopSentenceList) > leng:
				return descConnGraphTopSentenceList
	return descConnGraphTopSentenceList

def descConnGraphOneSupportAttackSentenceFind(topSentenceList,topicList,annotatedGraph,leng):
	finalList = set()
	G = annotatedGraph
	if int(leng) == 500:
		leng = 650
	if int(leng) == 1000:
		leng = 1100
	if int(leng) == 1500:
		leng = 1650
	for i,c in enumerate(sorted(nx.connected_component_subgraphs(annotatedGraph), key=len, reverse=True)):
		for s in c:
			attackDone = False
			supportDone = False
			for node1,node2 in c.edges(s):
				if not attackDone:
					if G[node1][node2]['relation'] == "attacks":
						if sentSimilarityWordWise(topSentenceList,G.nodes[node1]['sent'],topicList):
							finalList.add(G.nodes[node1]['sent'])
						if sentSimilarityWordWise(topSentenceList,G.nodes[node2]['sent'],topicList):
							finalList.add(G.nodes[node2]['sent'])
						attackDone = True
				if not supportDone:
					if G[node1][node2]['relation'] == "supports":
						if sentSimilarityWordWise(topSentenceList,G.nodes[node1]['sent'],topicList):
							finalList.add(G.nodes[node1]['sent'])
						if sentSimilarityWordWise(topSentenceList,G.nodes[node2]['sent'],topicList):
							finalList.add(G.nodes[node2]['sent'])
						supportDone = True
				if getWordCount(finalList) > leng:
						return finalList
				if supportDone and attackDone:
					break
	'''for c in sorted(nx.connected_components(annotatedGraph), key=len, reverse=True):
		for s in c:
			for node1,node2 in G.edges(s):
				finalList.add(G.nodes[node1]['sent'])
				finalList.add(G.nodes[node2]['sent'])
				if getWordCount(finalList) > leng:
					return finalList'''


def asceConnGraphOneSupportAttackSentenceFind(topSentenceList,topicList,annotatedGraph,leng):
	finalList = set()
	G = annotatedGraph
	if int(leng) == 500:
		leng = 550
	if int(leng) == 1000:
		leng = 1100
	if int(leng) == 1500:
		leng = 1650
	for c in sorted(nx.connected_components(annotatedGraph), key=len, reverse=False):
		for s in c:
			attackDone = False
			supportDone = False
			for node1,node2 in G.edges(s):
				if not attackDone:
					if G[node1][node2]['relation'] == "attacks":
						if sentSimilarityWordWise(topSentenceList,G.nodes[node1]['sent'],topicList):
							finalList.add(G.nodes[node1]['sent'])
						if sentSimilarityWordWise(topSentenceList,G.nodes[node2]['sent'],topicList):
							finalList.add(G.nodes[node2]['sent'])
						attackDone = True
				if not supportDone:
					if G[node1][node2]['relation'] == "supports":
						if sentSimilarityWordWise(topSentenceList,G.nodes[node1]['sent'],topicList):
							finalList.add(G.nodes[node1]['sent'])
						if sentSimilarityWordWise(topSentenceList,G.nodes[node2]['sent'],topicList):
							finalList.add(G.nodes[node2]['sent'])
						supportDone = True
				if getWordCount(finalList) > leng:
						return finalList
	return finalList

def descConnGraphOneSentenceFind(topSentenceList,topicList,annotatedGraph,leng):
	descConnGraphOneTopSentenceList = set()
	G = annotatedGraph
	if int(leng) == 500:
		leng = 550
	if int(leng) == 1000:
		leng = 1100
	if int(leng) == 1500:
		leng = 1650
	for c in sorted(nx.connected_components(annotatedGraph), key=len, reverse=True):
		count = 0
		for s in c:
			count = count + 1
			if sentSimilarityWordWise(topSentenceList,G.nodes[s]['sent'],topicList):
				descConnGraphOneTopSentenceList.add(G.nodes[s]['sent'])
			if getWordCount(descConnGraphOneTopSentenceList) > leng:
				return descConnGraphOneTopSentenceList
			if count == 1:
				break
	return descConnGraphOneTopSentenceList

def asceConnGraphOneSentenceFind(topSentenceList,topicList,annotatedGraph,leng):
	finalList = set()
	G = annotatedGraph
	if int(leng) == 500:
		leng = 550
	if int(leng) == 1000:
		leng = 1100
	if int(leng) == 1500:
		leng = 1650
	for c in sorted(nx.connected_components(annotatedGraph), key=len, reverse=False):
		count = 0
		for s in c:
			count = count + 1
			if sentSimilarityWordWise(topSentenceList,G.nodes[s]['sent'],topicList):
				finalList.add(G.nodes[s]['sent'])
			if getWordCount(finalList) > leng:
				return finalList
			if count == 1:
				break
	return finalList


def asceConnGraphSentenceFind(topSentenceList,topicList,annotatedGraph,leng):
	asceConnGraphTopSentenceList = set()
	G = annotatedGraph
	leng = int(leng) + 66
	for c in sorted(nx.connected_components(annotatedGraph), key=len):
		for s in c:
			if sentSimilarityWordWise(topSentenceList,G.nodes[s]['sent'],topicList):
				asceConnGraphTopSentenceList.add(G.nodes[s]['sent'])
			if getWordCount(asceConnGraphTopSentenceList) > leng:
				return asceConnGraphTopSentenceList
	return asceConnGraphTopSentenceList

def descDegreeSentenceFind(topSentenceList,topicList,annotatedGraph,leng):
	descDegreeSentenceList = set()
	G = annotatedGraph
	if int(leng) == 500:
		leng = 550
	if int(leng) == 1000:
		leng = 1100
	if int(leng) == 1500:
		leng = 1650
	degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)
	for degree in degrees:
		if sentSimilarityWordWise(topSentenceList,G.nodes[degree[0]]['sent'],topicList):
			descDegreeSentenceList.add(G.nodes[degree[0]]['sent'])
		if getWordCount(descDegreeSentenceList) > leng:
			return descDegreeSentenceList
	return descDegreeSentenceList

def asceDegreeSentenceFind(topSentenceList,topicList,annotatedGraph,leng):
	asceDegreeSentenceList = set()
	G = annotatedGraph
	if int(leng) == 500:
		leng = 550
	if int(leng) == 1000:
		leng = 1100
	if int(leng) == 1500:
		leng = 1650
	degrees = sorted(G.degree, key=lambda x: x[1])
	for degree in degrees:
		if sentSimilarityWordWise(topSentenceList,G.nodes[degree[0]]['sent'],topicList):
			asceDegreeSentenceList.add(G.nodes[degree[0]]['sent'])
		if getWordCount(asceDegreeSentenceList) > leng:
			return asceDegreeSentenceList
	return asceDegreeSentenceList

def intuitionSentenceFind(topSentenceList,topicList,annotatedGraph,leng):
	finalList = set()
	G = annotatedGraph
	if int(leng) == 500:
		leng = int(leng) + 50
	if int(leng) == 1000:
		leng = int(leng) + 100
	if int(leng) == 1500:
		leng = int(leng) + 150	
	for c in sorted(nx.connected_components(annotatedGraph), key=len, reverse=True):
		#print "---------------"
		for s in c:
			#print G.nodes[s]['sent']
			attackDone = False
			supportDone = False
			for node1,node2 in G.edges(s):   # Checking for attack
				if not attackDone:
					if G[node1][node2]['relation'] == "attacks":
						if sentSimilarityWordWise(topSentenceList,G.nodes[node1]['sent'],topicList):
							finalList.add(G.nodes[node1]['sent'])
						if sentSimilarityWordWise(topSentenceList,G.nodes[node2]['sent'],topicList):
							finalList.add(G.nodes[node2]['sent'])
						attackDone = True
				if not supportDone:
					if G[node1][node2]['relation'] == "supports":
						if sentSimilarityWordWise(topSentenceList,G.nodes[node1]['sent'],topicList):
							finalList.add(G.nodes[node1]['sent'])
						if sentSimilarityWordWise(topSentenceList,G.nodes[node2]['sent'],topicList):
							finalList.add(G.nodes[node2]['sent'])
						supportDone = True
				if getWordCount(finalList) > leng:
						return finalList
	return finalList


def evaluation(speechsList,claimDupTopSentenceList,debateId,leng,partner):
	#print "Running Evaluation - Rougue Scores"
	lengthOfWords = [int(leng)]
	
	for leng in lengthOfWords:
		# Gold Summary
		if partner:
			goldSummaryPathFile = goldSummary_Path + str(debateId) +  "_1" + "/" + str(leng) + ".txt"
		else:
			goldSummaryPathFile = goldSummary_Path + str(debateId) + "/" + str(leng) + ".txt"
		file = open(goldSummaryPathFile,"r") 
		summary = file.read()

		# Rouge
		rouge = Rouge()

		# TextRank Scores
		document = ""
		for speech in speechsList:
			document = document + speech + " "
		referenceTexRank = summarize(document,words=leng,language='english')
		#print referenceTexRank
		#print "--------------------------------------------------------------------------------------------------------------------------"
		#print getWordCount(referenceTexRank)
		scoresTextRank = rouge.get_scores(referenceTexRank,summary) # Check the order once TODO
		scores['textRank']['rouge-1']['p'] = scores['textRank']['rouge-1']['p'] + scoresTextRank[0]['rouge-1']['p']
		scores['textRank']['rouge-1']['r'] = scores['textRank']['rouge-1']['r'] + scoresTextRank[0]['rouge-1']['r']
		scores['textRank']['rouge-1']['f'] = scores['textRank']['rouge-1']['f'] + scoresTextRank[0]['rouge-1']['f']
		scores['textRank']['rouge-2']['p'] = scores['textRank']['rouge-2']['p'] + scoresTextRank[0]['rouge-2']['p']
		scores['textRank']['rouge-2']['r'] = scores['textRank']['rouge-2']['r'] + scoresTextRank[0]['rouge-2']['r']
		scores['textRank']['rouge-2']['f'] = scores['textRank']['rouge-2']['f'] + scoresTextRank[0]['rouge-2']['f']
		scores['textRank']['rouge-l']['p'] = scores['textRank']['rouge-l']['p'] + scoresTextRank[0]['rouge-l']['p']
		scores['textRank']['rouge-l']['r'] = scores['textRank']['rouge-l']['r'] + scoresTextRank[0]['rouge-l']['r']
		scores['textRank']['rouge-l']['f'] = scores['textRank']['rouge-l']['f'] + scoresTextRank[0]['rouge-l']['f']
		
		# Nenkova Scores
		os.system("python summarizer_topic.py " + str(debateId) + ".txt" + " " + str(leng) + " " + "> nenkova.txt")
		fileNenkova = open("nenkova.txt","r")
		referenceNenkova = fileNenkova.read()
		#print referenceNenkova
		#print "--------------------------------------------------------------------------------------------------------------------------"
		#scoresNenkova = rouge.get_scores(summary,referenceNenkova) # Check the order once TODO
		scoresNenkova = rouge.get_scores(referenceNenkova,summary) 
		scores['nenkova']['rouge-1']['p'] = scores['nenkova']['rouge-1']['p'] + scoresNenkova[0]['rouge-1']['p']
		scores['nenkova']['rouge-1']['r'] = scores['nenkova']['rouge-1']['r'] + scoresNenkova[0]['rouge-1']['r']
		scores['nenkova']['rouge-1']['f'] = scores['nenkova']['rouge-1']['f'] + scoresNenkova[0]['rouge-1']['f']
		scores['nenkova']['rouge-2']['p'] = scores['nenkova']['rouge-2']['p'] + scoresNenkova[0]['rouge-2']['p']
		scores['nenkova']['rouge-2']['r'] = scores['nenkova']['rouge-2']['r'] + scoresNenkova[0]['rouge-2']['r']
		scores['nenkova']['rouge-2']['f'] = scores['nenkova']['rouge-2']['f'] + scoresNenkova[0]['rouge-2']['f']
		scores['nenkova']['rouge-l']['p'] = scores['nenkova']['rouge-l']['p'] + scoresNenkova[0]['rouge-l']['p']
		scores['nenkova']['rouge-l']['r'] = scores['nenkova']['rouge-l']['r'] + scoresNenkova[0]['rouge-l']['r']
		scores['nenkova']['rouge-l']['f'] = scores['nenkova']['rouge-l']['f'] + scoresNenkova[0]['rouge-l']['f']
		#print getWordCount(referenceNenkova)

		'''# LexRank Scores
		parser = PlaintextParser.from_string(document, Tokenizer("english"))
		summarizer = LexRankSummarizer()
		referenceLexRank = ""
		if int(leng) == 500:
			length = 30
		if int(leng) == 1000:
			length = 70
		if int(leng) == 1500:
			length = 120
		while 1:
			summaryLexRank = summarizer(parser.document, int(length))
			referenceLexRank = ""
			for sentence in summaryLexRank:	
				referenceLexRank = referenceLexRank + str(sentence) + " "
			if getWordCount(referenceLexRank) > int(leng):
				break
			length = length + 5
		scoresLexRank = rouge.get_scores(referenceLexRank,summary) 
		scores['lexRank']['rouge-1']['p'] = scores['lexRank']['rouge-1']['p'] + scoresLexRank[0]['rouge-1']['p']
		scores['lexRank']['rouge-1']['r'] = scores['lexRank']['rouge-1']['r'] + scoresLexRank[0]['rouge-1']['r']
		scores['lexRank']['rouge-1']['f'] = scores['lexRank']['rouge-1']['f'] + scoresLexRank[0]['rouge-1']['f']
		scores['lexRank']['rouge-2']['p'] = scores['lexRank']['rouge-2']['p'] + scoresLexRank[0]['rouge-2']['p']
		scores['lexRank']['rouge-2']['r'] = scores['lexRank']['rouge-2']['r'] + scoresLexRank[0]['rouge-2']['r']
		scores['lexRank']['rouge-2']['f'] = scores['lexRank']['rouge-2']['f'] + scoresLexRank[0]['rouge-2']['f']
		scores['lexRank']['rouge-l']['p'] = scores['lexRank']['rouge-l']['p'] + scoresLexRank[0]['rouge-l']['p']
		scores['lexRank']['rouge-l']['r'] = scores['lexRank']['rouge-l']['r'] + scoresLexRank[0]['rouge-l']['r']
		scores['lexRank']['rouge-l']['f'] = scores['lexRank']['rouge-l']['f'] + scoresLexRank[0]['rouge-l']['f']
		print getWordCount(referenceLexRank)'''

		# Own
		referenceOwn = ""
		for sent in claimDupTopSentenceList:
			referenceOwn = referenceOwn + sent + " "
		print(referenceOwn)
		print("----------------------------------------------------------------------------------------------------------------------------")
		scoresOwn = rouge.get_scores(referenceOwn,summary)
		scores['own']['rouge-1']['p'] = scores['own']['rouge-1']['p'] + scoresOwn[0]['rouge-1']['p']
		scores['own']['rouge-1']['r'] = scores['own']['rouge-1']['r'] + scoresOwn[0]['rouge-1']['r']
		scores['own']['rouge-1']['f'] = scores['own']['rouge-1']['f'] + scoresOwn[0]['rouge-1']['f']
		scores['own']['rouge-2']['p'] = scores['own']['rouge-2']['p'] + scoresOwn[0]['rouge-2']['p']
		scores['own']['rouge-2']['r'] = scores['own']['rouge-2']['r'] + scoresOwn[0]['rouge-2']['r']
		scores['own']['rouge-2']['f'] = scores['own']['rouge-2']['f'] + scoresOwn[0]['rouge-2']['f']
		scores['own']['rouge-l']['p'] = scores['own']['rouge-l']['p'] + scoresOwn[0]['rouge-l']['p']
		scores['own']['rouge-l']['r'] = scores['own']['rouge-l']['r'] + scoresOwn[0]['rouge-l']['r']
		scores['own']['rouge-l']['f'] = scores['own']['rouge-l']['f'] + scoresOwn[0]['rouge-l']['f']
		print(getWordCount(referenceOwn))


	# Calculate F-score with Rouge 1,2,L #https://rxnlp.com/how-rouge-works-for-evaluation-of-summarization-tasks/#.W9l_DXUzbCI

def makeOriginalDebateFile(sentenceList,debateId):
	f = open(debateId + ".txt","w+")
	for sent in sentenceList:
		f.write(sent + "\n")
	f.close()

def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'
 
    if tag.startswith('V'):
        return 'v'
 
    if tag.startswith('J'):
        return 'a'
 
    if tag.startswith('R'):
        return 'r'
 
    return None
 
def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
 
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None
 
def sentence_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))
 
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
 
    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]
 
    score, count = 0.0, 0
 
    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        best_score = max([synset.path_similarity(ss) for ss in synsets2])
 
        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1
 
    # Average the values
    score /= count
    return score

def mergeIdentical(G):
	checkNode = set()
	for n1,v1 in G.nodes(data=True):
		for n2,v2 in G.nodes(data=True):
			if v1['id'] != v2['id']:
				if n1 not in checkNode and n2 not in checkNode:
					if v1['start'] == v2['start'] and v1['end'] == v2['end']:
						checkNode.add(n1)
						checkNode.add(n2)
						if n1 > n2:
							v2['identical'] = n1
						else:
							v1['identical'] = n2
	return None

def makeGraphOfAnnotation(debateId):
	annotationFile = ""
	try:
		annotationFile = open("../AnnotationDebates/brat-v1.3_Crunchy_Frog/data/debates/svkrohit/" + debateId + ".ann","r")
	except:
		annotationFile = open("../AnnotationDebates/brat-v1.3_Crunchy_Frog/data/debates/" + debateId + ".ann","r")
	G = nx.Graph()
	G.add_node("root",id="",sent="",entity="",start="",end="")
	vertexCount = 0
	for line in annotationFile: # vertex count starts from 1
		splittedLine = line.split("\t")
		if "Claim" in line:
			vertexCount = vertexCount + 1
			G.add_node(vertexCount,id=splittedLine[0],sent=splittedLine[2],entity="Claim",start=splittedLine[1].split(" ")[1],end=splittedLine[1].split(" ")[2])
			#G.add_edge("root",vertexCount)
		if "Premise" in line:
			vertexCount = vertexCount + 1
			G.add_node(vertexCount,id=splittedLine[0],sent=splittedLine[2],entity="Premise",start=splittedLine[1].split(" ")[1],end=splittedLine[1].split(" ")[2])
			#G.add_edge("root",vertexCount)
		mergeIdentical(G)
		if "supports" in line:
			claimId = splittedLine[1].split(" ")[2].split(":")[1]
			premiseId = splittedLine[1].split(" ")[1].split(":")[1]
			nodeClaim = [n for n,v in G.nodes(data=True) if v['id'] == claimId]   
			nodePremise = [n for n,v in G.nodes(data=True) if v['id'] == premiseId]
			try:
				nodeClaim = list(G.nodes()[nodeClaim[0]]['identical'])
			except:
				pass
			try:
				nodePremise = list(G.nodes()[nodePremise[0]]['identical'])
			except:
				pass
			G.add_edge(nodeClaim[0],nodePremise[0],relation="supports")
		if "attacks" in line and "Arg" in line:
			claimId = splittedLine[1].split(" ")[2].split(":")[1]
			premiseId = splittedLine[1].split(" ")[1].split(":")[1]
			nodeClaim = [n for n,v in G.nodes(data=True) if v['id'] == claimId]   
			nodePremise = [n for n,v in G.nodes(data=True) if v['id'] == premiseId]
			G.add_edge(nodeClaim[0],nodePremise[0],relation="attacks")


	'''for i,c in enumerate(sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)):
		for s in c:
			attackDone = False
			supportDone = False
			for node1,node2 in c.edges(s):
				if c[node1][node2]['relation'] == "attacks":
					attackDone = True
				if c[node1][node2]['relation'] == "supports":
					supportDone = True
			if attackDone and supportDone:
				# Draw graph
				pos = nx.spring_layout(c,scale=100)
				plt.figure(10)
				nx.draw(c,pos,node_size=3260,font_size=5)
				node_labels = nx.get_node_attributes(c,'sent')
				nx.draw_networkx_labels(c,pos,labels = node_labels)
				edge_labels = nx.get_edge_attributes(c,'relation') 
				#edge_labels = {e: i['relation'] for i in c.edges(data=True)}
				nx.draw_networkx_edge_labels(c, pos, labels = edge_labels)
				plt.show()'''
	return G

def runPipeline(caseCheck,topSentenceList,topicList,annotatedGraph,leng):
	if caseCheck == "claimPremiseSupport":
		claimPremiseSupportTopSentenceList = claimPremiseSupportSentenceFind(topSentenceList,topicList,annotatedGraph,leng)
		#print "Total Number of Sentences After Logic : " + str(len(claimPremiseSupportTopSentenceList))
		return claimPremiseSupportTopSentenceList
	elif caseCheck == "claimOnePremise":
		claimOnePremiseTopSentenceList = claimOnePremiseSentenceFind(topSentenceList,topicList,annotatedGraph,leng)
		#print "Total Number of Sentences After Logic : " + str(len(claimOnePremiseTopSentenceList))
		return claimOnePremiseTopSentenceList
	elif caseCheck == "claimTwoPremise":
		claimTwoPremiseTopSentenceList = claimTwoPremiseSentenceFind(topSentenceList,topicList,annotatedGraph,leng)
		#print "Total Number of Sentences After Logic : " + str(len(claimTwoPremiseTopSentenceList))
		return claimTwoPremiseTopSentenceList
	elif caseCheck == "claimThreePremise":
		claimThreePremiseTopSentenceList = claimThreePremiseSentenceFind(topSentenceList,topicList,annotatedGraph,leng)
		#print "Total Number of Sentences After Logic : " + str(len(claimThreePremiseTopSentenceList))
		return claimThreePremiseTopSentenceList
	elif caseCheck == "descConnGraph":
		descConnGraphTopSentenceList = descConnGraphSentenceFind(topSentenceList,topicList,annotatedGraph,leng)
		#print "Total Number of Sentences After Logic : " + str(len(descConnGraphTopSentenceList))
		return descConnGraphTopSentenceList
	elif caseCheck == "asceConnGraph":
		asceConnGraphTopSentenceList = asceConnGraphSentenceFind(topSentenceList,topicList,annotatedGraph,leng)
		#print "Total Number of Sentences After Logic : " + str(len(asceConnGraphTopSentenceList))
		return asceConnGraphTopSentenceList
	elif caseCheck == "descConnGraphOne":
		descConnGraphTopSentenceList = descConnGraphOneSentenceFind(topSentenceList,topicList,annotatedGraph,leng)
		#print "Total Number of Sentences After Logic : " + str(len(descConnGraphTopSentenceList))
		return descConnGraphTopSentenceList
	elif caseCheck == "asceConnGraphOne":
		asceConnGraphTopSentenceList = asceConnGraphOneSentenceFind(topSentenceList,topicList,annotatedGraph,leng)
		#print "Total Number of Sentences After Logic : " + str(len(asceConnGraphTopSentenceList))
		return asceConnGraphTopSentenceList
	elif caseCheck =="descConnGraphOneAttackSupport":
		descConnGraphOneTopSentenceList = descConnGraphOneSupportAttackSentenceFind(topSentenceList,topicList,annotatedGraph,leng)
		#print "Total Number of Sentences After Logic : " + str(len(descConnGraphOneTopSentenceList))
		return descConnGraphOneTopSentenceList
	elif caseCheck =="asceConnGraphOneAttackSupport":
		asceConnGraphOneTopSentenceList = asceConnGraphOneSupportAttackSentenceFind(topSentenceList,topicList,annotatedGraph,leng)
		#print "Total Number of Sentences After Logic : " + str(len(asceConnGraphOneTopSentenceList))
		return asceConnGraphOneTopSentenceList
	elif caseCheck == "descDegree":
		descDegreeSentenceList = descDegreeSentenceFind(topSentenceList,topicList,annotatedGraph,leng)
		#print "Total Number of Sentences After Logic : " + str(len(descDegreeSentenceList))
		return descDegreeSentenceList
	elif caseCheck == "asceDegree":
		asceDegreeSentenceList = asceDegreeSentenceFind(topSentenceList,topicList,annotatedGraph,leng)
		#print "Total Number of Sentences After Logic : " + str(len(asceDegreeSentenceList))
		return asceDegreeSentenceList
	elif caseCheck == "intuition":
		intuitionSentenceList = intuitionSentenceFind(topSentenceList,topicList,annotatedGraph,leng)
		#print "Total Number of Sentences After Logic : " + str(len(intuitionSentenceList))
		return intuitionSentenceList

def printEvaluation():
	print("   " + str(leng)  + "                Rouge 1                                      Rouge 2                                             Rouge L")
	print("TextRank " + str(scores['textRank']['rouge-1']['p']) + " " + str(scores['textRank']['rouge-1']['r']) + " " + str(scores['textRank']['rouge-1']['f']) + "  " + str(scores['textRank']['rouge-2']['p']) + " " + str(scores['textRank']['rouge-2']['r']) + " " + str(scores['textRank']['rouge-2']['f']) + "  " + str(scores['textRank']['rouge-l']['p']) + " " + str(scores['textRank']['rouge-l']['r']) + " " + str(scores['textRank']['rouge-l']['f']))
	print("LexRank  " + str(scores['lexRank']['rouge-1']['p']) + " " + str(scores['lexRank']['rouge-1']['r']) + " " + str(scores['lexRank']['rouge-1']['f']) + "  " + str(scores['lexRank']['rouge-2']['p']) + " " + str(scores['lexRank']['rouge-2']['r']) + " " + str(scores['lexRank']['rouge-2']['f']) + "  " + str(scores['lexRank']['rouge-l']['p']) + " " + str(scores['lexRank']['rouge-l']['r']) + " " + str(scores['lexRank']['rouge-l']['f']))
	print("Nenkova  " + str(scores['nenkova']['rouge-1']['p']) + " " + str(scores['nenkova']['rouge-1']['r']) + " " + str(scores['nenkova']['rouge-1']['f']) + "  " + str(scores['nenkova']['rouge-2']['p']) + " " + str(scores['nenkova']['rouge-2']['r']) + " " + str(scores['nenkova']['rouge-2']['f']) + "  " + str(scores['nenkova']['rouge-l']['p']) + " " + str(scores['nenkova']['rouge-l']['r']) + " " + str(scores['nenkova']['rouge-l']['f']))
	print("Own      " + str(scores['own']['rouge-1']['p']) + " " + str(scores['own']['rouge-1']['r']) + " " + str(scores['own']['rouge-1']['f']) + "  " + str(scores['own']['rouge-2']['p']) + " " + str(scores['own']['rouge-2']['r']) + " " + str(scores['own']['rouge-2']['f']) + "  " + str(scores['own']['rouge-l']['p']) + " " + str(scores['own']['rouge-l']['r']) + " " + str(scores['own']['rouge-l']['f']))
	print("\n")
	print("\n")

	scores['textRank'] = {}
	scores['textRank']['rouge-1'] = {}
	scores['textRank']['rouge-2'] = {}
	scores['textRank']['rouge-l'] = {}
	scores['textRank']['rouge-1']['p'] = 0
	scores['textRank']['rouge-1']['r'] = 0
	scores['textRank']['rouge-1']['f'] = 0
	scores['textRank']['rouge-2']['p'] = 0
	scores['textRank']['rouge-2']['r'] = 0
	scores['textRank']['rouge-2']['f'] = 0
	scores['textRank']['rouge-l']['p'] = 0
	scores['textRank']['rouge-l']['r'] = 0
	scores['textRank']['rouge-l']['f'] = 0
	scores['nenkova'] = {}
	scores['nenkova']['rouge-1'] = {}
	scores['nenkova']['rouge-2'] = {}
	scores['nenkova']['rouge-l'] = {}
	scores['nenkova']['rouge-1']['p'] = 0
	scores['nenkova']['rouge-1']['r'] = 0
	scores['nenkova']['rouge-1']['f'] = 0
	scores['nenkova']['rouge-2']['p'] = 0
	scores['nenkova']['rouge-2']['r'] = 0
	scores['nenkova']['rouge-2']['f'] = 0
	scores['nenkova']['rouge-l']['p'] = 0
	scores['nenkova']['rouge-l']['r'] = 0
	scores['nenkova']['rouge-l']['f'] = 0
	scores['lexRank'] = {}
	scores['lexRank']['rouge-1'] = {}
	scores['lexRank']['rouge-2'] = {}
	scores['lexRank']['rouge-l'] = {}
	scores['lexRank']['rouge-1']['p'] = 0
	scores['lexRank']['rouge-1']['r'] = 0
	scores['lexRank']['rouge-1']['f'] = 0
	scores['lexRank']['rouge-2']['p'] = 0
	scores['lexRank']['rouge-2']['r'] = 0
	scores['lexRank']['rouge-2']['f'] = 0
	scores['lexRank']['rouge-l']['p'] = 0
	scores['lexRank']['rouge-l']['r'] = 0
	scores['lexRank']['rouge-l']['f'] = 0
	scores['own'] = {}
	scores['own']['rouge-1'] = {}
	scores['own']['rouge-2'] = {}
	scores['own']['rouge-l'] = {}
	scores['own']['rouge-1']['p'] = 0
	scores['own']['rouge-1']['r'] = 0
	scores['own']['rouge-1']['f'] = 0
	scores['own']['rouge-2']['p'] = 0
	scores['own']['rouge-2']['r'] = 0
	scores['own']['rouge-2']['f'] = 0
	scores['own']['rouge-l']['p'] = 0
	scores['own']['rouge-l']['r'] = 0
	scores['own']['rouge-l']['f'] = 0

def removeSmallSent(topSentenceList):
	finalList = list()
	for sent in topSentenceList:
		if len(sent.split()) >= 5:
			finalList.append(sent)
	return finalList

def removeSentiScore(topSentenceList):
	finalList = list()
	sid = SIA()
	sent = 0.0
	for sent in topSentenceList:
		ss = sid.polarity_scores(sent)
		if abs(ss['compound']) > 0.5:
			finalList.append(sent)
	return finalList
		#print ss
		#print sent

if __name__ == "__main__":
	tempList = ["Therefore, Jammu & Kashmir, under their Constitution, had to separately undergo a legislative process.","Jammu & Kashmir went through that legislative process and finally passed a Resolution in their State Assembly"]
	#print removeDuplicateSent(tempList)
	#for pipeline in ["claimPremiseSupport","claimOnePremise","claimTwoPremise","claimThreePremise","descConnGraph","asceConnGraph","descConnGraphOne","asceConnGraphOne","descConnGraphOneAttackSupport","asceConnGraphOneAttackSupport"]:
	for pipeline in ["descConnGraphOneAttackSupport"]:
		print("Pipeline " + pipeline)
		#for model in ["TextRank","LSA","LDA","NMF"]:
		for model in ["NMF"]:
			print("Topic Model " + model)
			topicModel = model
			for leng in [500]:
				print("Length      " + str(leng))
				for debateId in debatesList:
					print("Debate      " + str(debateId))
					partner = False
					if "_1" in debateId:
						partner = True
						debateId = debateId.split("_1")[0]
					speechsList = debateFind(debateId)
					sentenceList = sentenceFind(speechsList)
					makeOriginalDebateFile(sentenceList,debateId)	
					#print "Total Number of Sentences : " + str(len(sentenceList))
					preprocessedSpeechList = stemStopWordPOSSpeeches(speechsList)
					topicList = topicFind(preprocessedSpeechList,topicModel,leng)
					topSentenceList = topSentenceFind(topicList,sentenceList)
					#print "Total Number of Sentences After Topics : " + str(len(topSentenceList))
					annotatedGraph = makeGraphOfAnnotation(debateId)
					
					#ipeline(pipeline,topSentenceList,topicList,annotatedGraph,leng,debateId,partner)
					#print str(len(topSentenceList))
					finalList = removeSmallSent(topSentenceList)
					finalList = removeSentiScore(topSentenceList) 

					#print str(len(finalList))
					finalList = removeDuplicateSent(finalList)
					finalList = runPipeline(pipeline,finalList,topicList,annotatedGraph,leng)
					finalList = removeDuplicateSent(finalList)
					#print "Total Number of Sentences Final : " + str(len(finalList))

					#print "Total Number of Sentences After Duplication Removal : " + str(len(finalList))
				    
				    # Evaluation
					evaluation(speechsList,finalList,debateId,leng,partner)

				printEvaluation()
