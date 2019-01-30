from rouge import Rouge
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import gensim
import os
import collections
import smart_open
import random
from main import NMF_run


#debatesList = ["5a42549689d32a43eb059cff","5a42537189d32a43eb04e6c7","5a42653e89d32a43eb1034cf","5a4257a889d32a43eb07987b","5a426fb289d32a43eb15d389"]
debatesList = ["5a42549689d32a43eb059cff"]

def irr():
	for leng in [1000,1500]:
		for debateId in debatesList:
			file1 = open("./goldSummary/" + debateId + "/" + str(leng) + ".txt","r")
			file2 = open("./goldSummary/" + debateId + "_1/" +  str(leng) +  ".txt","r")
			text1 = file1.read()
			text2 = file2.read()
			rouge = Rouge()
			print rouge.get_scores(text1,text2)

def nmf():
	for leng in [1000,1500]:
		for debateId in debatesList:
			file1 = open("./goldSummary/" + debateId + "/" + str(leng) + ".txt","r")
			file2 = open("./goldSummary/" + debateId + "_1/" +  str(leng) +  ".txt","r")
			text1 = file1.read()
			text2 = file2.read()
			print NMF_run(text1)
			print NMF_run(text2)

if __name__ == "__main__":
	nmf()