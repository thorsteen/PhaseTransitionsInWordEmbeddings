# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 15:42:20 2020

@author: Thor Steen Larsen
"""
import nltk
from nltk.corpus import wordnet   #Import wordnet from the NLTK
import pandas as pd
import sys
import os

"""
#test
synset = wordnet.synsets("Travel")
print('Word and Type : ' + synset[0].name())
print('Synonym of Travel is: ' + synset[0].lemmas()[0].name())
print('The meaning of the word : ' + synset[0].definition())
print('Example of Travel : ' + str(synset[0].examples()))
"""


def find_all_semantics(word):
    syn = list()
    ant = list()
    hypernym = list()
    hyponym = list()
    member_holonym = list()
    root_hypernym = list()
    for synset in wordnet.synsets(word):
        for hyper in synset.hypernyms():
            hypernym.append(hyper.name())
        for hypo in synset.hyponyms():
            hyponym.append(hypo.name())
        for members in synset.member_holonyms():
            member_holonym.append(members.name())
        for roots in synset.root_hypernyms():
            root_hypernym.append(roots.name())
        for lemma in synset.lemmas():
            syn.append(lemma.name())    #add the synonyms
            if lemma.antonyms():    #When antonyms are available, add them into the list
              ant.append(lemma.antonyms()[0].name())
    print('Synonyms: ' + str(syn))
    print('Antonyms: ' + str(ant))
    print('Hypernyms: ' + str(hypernym))
    print('Root Hypernyms: '+str(root_hypernym))
    print('Hyponyms: ' + str(hyponym))
    print('Member Holonyms: '+str(member_holonym))


results = pd.read_excel("model_word_pair_comparison_ws353.xlsx")
#remove missing
results.drop(results[results.cbow_model_w1 == 'missing'].index,inplace = True)

#we want to compare to ws353 ranking
ws_353_scores = results.sort_values("WS_similiarity_score",ascending=False, ignore_index=True)
#sort in decedning order for first sim score (cosine sim)
results.sort_values('cbow_model_w1', ascending=False, ignore_index=True, inplace = True)
#write results from wordnet to text file
orig_stdout = sys.stdout
os.remove("CBoW_W1_word_pair_ws353_semantics.txt")
f = open('CBoW_W1_word_pair_ws353_semantics.txt', 'w')
sys.stdout = f

for i in range(len(results)):
    print("Rank {} word pair ({},{}) w/ cosine similarity = {} (WS353 rank = {})".format(i+1,results.iloc[i,0],results.iloc[i,1],results.iloc[i,2],ws_353_scores[ws_353_scores['word_0']==results.iloc[i,0]].index[0]))
    print("-----------------------------------------")
    print("Semantic relations for "+results.iloc[i,0])
    print("")
    find_all_semantics(results.iloc[i,0])
    print("")
    print("Semantic relations for "+results.iloc[i,1])
    print("")
    find_all_semantics(results.iloc[i,1])
    print("")
    print("")

sys.stdout = orig_stdout
f.close()

results.sort_values('cbow_model_w2', ascending=False, ignore_index=True, inplace = True)
#write results from wordnet to text file
orig_stdout = sys.stdout
os.remove("CBoW_W2_word_pair_ws353_semantics.txt")
f = open('CBoW_W2_word_pair_ws353_semantics.txt', 'w')
sys.stdout = f

for i in range(len(results)):
    print("Rank {} word pair ({},{}) w/ cosine similarity = {} (WS353 rank = {})".format(i+1,results.iloc[i,0],results.iloc[i,1],results.iloc[i,3],ws_353_scores[ws_353_scores['word_0']==results.iloc[i,0]].index[0]))
    print("-----------------------------------------")
    print("Semantic relations for "+results.iloc[i,0])
    print("")
    find_all_semantics(results.iloc[i,0])
    print("")
    print("Semantic relations for "+results.iloc[i,1])
    print("")
    find_all_semantics(results.iloc[i,1])
    print("")
    print("")

sys.stdout = orig_stdout
f.close()

results.sort_values('cbow_model_w25', ascending=False, ignore_index=True, inplace = True)
os.remove("CBoW_W25_word_pair_ws353_semantics.txt")
f = open('CBoW_W25_word_pair_ws353_semantics.txt', 'w')
sys.stdout = f

for i in range(len(results)):
    print("Rank {} word pair ({},{}) w/ cosine similarity = {} (WS353 rank = {})".format(i+1,results.iloc[i,0],results.iloc[i,1],results.iloc[i,27-1],ws_353_scores[ws_353_scores['word_0']==results.iloc[i,0]].index[0]))
    print("-----------------------------------------")
    print("Semantic relations for "+results.iloc[i,0])
    print("")
    find_all_semantics(results.iloc[i,0])
    print("")
    print("Semantic relations for "+results.iloc[i,1])
    print("")
    find_all_semantics(results.iloc[i,1])
    print("")
    print("")

sys.stdout = orig_stdout
f.close()
