# import modules & set up logging
import gensim, logging
import numpy as np
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

os.chdir(os.getcwd()+"\Documents\BaProject\PhaseTransitionsInWordEmbeddings\models")

sentences = gensim.models.word2vec.Text8Corpus("../Data/text8")

max_window_size = 25


for i in range(1, max_window_size+1):
    
    model = gensim.models.Word2Vec(sentences, min_count=500, size=500, workers=6, window = i, sg = 0, cbow_mean = 0) 
    #Training algorithm: 1 for skip-gram; otherwise CBOW.
    #cbow_mean ({0, 1}, optional) – If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
    
    model.save('Gensim_word2vec_models/cbow_model_w{}'.format(i))



for i in range(1, max_window_size+1):
    
    model = gensim.models.Word2Vec(sentences, min_count=500, size=500, workers=6, window = i, sg = 1) 
    #Training algorithm: 1 for skip-gram; otherwise CBOW.
    #cbow_mean ({0, 1}, optional) – If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
    
    model.save('Gensim_word2vec_models/SGNS_model_w{}'.format(i))



#for similiarity score
#https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.similarity
#model.most_similar('word')
#model.doesnt_match('word')
#model.similarity('word0', 'word1')

#to get vectors of a words
#model['word']


"""
Test of model word similarity function
"""

os.chdir(os.getcwd()+"\Gensim_word2vec_models")

virus_cbow = []
virus_sgns = []


file = writtenByFile = open("virus_word_pair_comparison.txt", "w+", encoding="utf-8")

try:
    for win in range(max_window_size):
        virus_cbow.append(('cbow_model_w'+str(win+1),gensim.models.Word2Vec.load('cbow_model_w'+str(win+1)).wv.most_similar("virus")))
        virus_sgns.append(('SGNS_model_w'+str(win+1),gensim.models.Word2Vec.load('SGNS_model_w'+str(win+1)).wv.most_similar("virus")))
        file.write("%s\n" % (str(virus_cbow[win])))
        file.write("%s\n" % (str(virus_sgns[win])))
except:
    print("Error: models cannot calculate similarity")

file.close()
