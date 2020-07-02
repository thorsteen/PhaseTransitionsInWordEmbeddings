# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 15:42:20 2020

@author: Thor Steen Larsen
"""
import nltk
from nltk.corpus import wordnet   #Import wordnet from the NLTK
#doku https://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html
import pandas as pd
import sys
import os

home = os.path.expanduser("~")
os.chdir(home + "\Documents\BaProject\PhaseTransitionsInWordEmbeddings\Data")

"""
#test
synset = wordnet.synsets("Travel")
print('Word and Type : ' + synset[0].name())
print('Synonym of Travel is: ' + synset[0].lemmas()[0].name())
print('The meaning of the word : ' + synset[0].definition())
print('Example of Travel : ' + str(synset[0].examples()))
"""


def find_all_semantics(word,verbose = True):
    syn = list()
    hypernym = list()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            syn.append(lemma.name())    #add the synonyms
        hypernym.append(sorted([lemma.name() for synset in synset.hyponyms() for lemma in synset.lemmas()]))
        
    if verbose:
        print('Synonyms of {}: '.format(word) + str(syn))
        print('Hypernyms of {}: '.format(word) + str(hypernym[0]))
    
    return syn, hypernym


def find_common_hypernyms(word1, word2, verbose = True):
    word1_syn = wordnet.synsets(word1)
    word2_syn = wordnet.synsets(word2)
    
    no = 3 #number of synsets to check
    
    if ((len(word1_syn) < 3) and (len(word2_syn) < 3)):
        no = 1
        
    mod = len(word1_syn) - (len(word1_syn) % len(word2_syn))
    
    print(no)
    
    print(word1_syn)
    print(word2_syn)
    
    common = []
    
    for i in range(no):
        common.append(word1_syn[i].common_hypernyms(word2_syn[i]))
    
    if verbose:
        co = []
        for com in common:
            for c in com:
                co.append(c.name())
        print(co)
        
        return co
    
    else:

        return common


"""
Path Distance Similarity:
        Return a score denoting how similar two word senses are, based on the
        shortest path that connects the senses in the is-a (hypernym/hypnoym)
        taxonomy. The score is in the range 0 to 1, except in those cases where
        a path cannot be found (will only be true for verbs as there are many
        distinct verb taxonomies), in which case None is returned. A score of
        1 represents identity i.e. comparing a sense with itself will return 1.
"""

def find_path_sim(word1, word2, verbose = True):
    word1_syn = wordnet.synsets(word1)
    word2_syn = wordnet.synsets(word2)
    word1 = word1_syn[0]
    word2 = word2_syn[0]
    sim = word1.path_similarity(word2)
    if verbose:
        print(sim)

    return sim
    

find_all_semantics('test')
find_all_semantics('trial')   
find_common_hypernyms("test","trail")
find_path_sim("test","trail")

results = pd.read_excel("model_word_pair_comparison_ws353.xlsx")
#remove missing
results.drop(results[results.cbow_model_w1 == 'missing'].index,inplace = True)

#we want to compare to ws353 ranking
ws_353_scores = results.sort_values("WS_similiarity_score",ascending=False, ignore_index=True)
#sort in decedning order for first sim score (cosine sim)
results.sort_values('cbow_model_w1', ascending=False, ignore_index=True, inplace = True)
#write results from wordnet to text file

f = open('CBoW_W1_word_pair_ws353_semantics.csv', 'w')

f.write("WordPair^PathSim^ModelSim^ModelRank^WS353Rank \n")

for i in range(len(results)):
    f.write("({},{})^{}^{}^{}^{} \n".format(results.iloc[i,0],results.iloc[i,1],find_path_sim(results.iloc[i,0],results.iloc[i,1],False),results.iloc[i,2],i+1,ws_353_scores[ws_353_scores['word_0']==results.iloc[i,0]].index[0]))
    
f.close()

orig_stdout = sys.stdout
f = open('CBoW_W1_word_pair_hyponyms.txt', 'w')
sys.stdout = f

for i in range(len(results)):
    print("Rank {} word pair ({},{}) w/ cosine similarity = {} (WS353 rank = {})".format(i+1,results.iloc[i,0],results.iloc[i,1],results.iloc[i,2],ws_353_scores[ws_353_scores['word_0']==results.iloc[i,0]].index[0]))
    print("-----------------------------------------")
    print("")
    find_all_semantics(results.iloc[i,0]) 
    print("")
    find_all_semantics(results.iloc[i,1]) 
    print("")
    print("")
    print("")

sys.stdout = orig_stdout
f.close()
