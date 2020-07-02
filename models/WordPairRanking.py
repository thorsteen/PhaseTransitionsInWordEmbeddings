# import modules & set up logging
import gensim, logging
import numpy as np
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

home = os.path.expanduser("~")
os.chdir(home+"\Documents\BaProject\PhaseTransitionsInWordEmbeddings\models")

max_window_size = 25

WS353File = '../../Data/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt'
SimLexFile = '../../Data/SimLex-999/SimLex-999-1.txt'


os.chdir(os.getcwd()+"\Gensim_word2vec_models")

def eval_wordpair_rank(file, max_window_size, wordpair_filename, usecols = (0,1,2)):
    wordpairs = np.loadtxt(wordpair_filename, dtype = str)
    
    file.write("Word1,Word2,")
    for win in range(max_window_size): file.write("CBoW_Model_W{},".format(win+1))
    for win in range(max_window_size): file.write("SGNS_Model_W{},".format(win+1))
    file.write("Reference_val \n")
    
    for wordpair in zip(wordpairs[:,0], wordpairs[:,1],wordpairs[:,2]):
        cosine_cbow = []
        cosine_sgns = []
        try:
            for win in range(max_window_size):
                cosine_cbow.append(('cbow_model_w'+str(win+1),gensim.models.Word2Vec.load('cbow_model_w'+str(win+1)).wv.similarity(wordpair[0],wordpair[1])))
                cosine_sgns.append(('SGNS_model_w'+str(win+1),gensim.models.Word2Vec.load('SGNS_model_w'+str(win+1)).wv.similarity(wordpair[0],wordpair[1])))
            
            file.write("%s," % wordpair[0])
            file.write("%s," % wordpair[1])
    
            for win in range(max_window_size): file.write("%s," % (str(cosine_cbow[win][1])))
            for win in range(max_window_size): file.write("%s," % (str(cosine_sgns[win][1])))
            file.write("%s" % wordpair[2])
            file.write("\n")
        except:
            file.write("%s," % wordpair[0])
            file.write("%s," % wordpair[1])
    
            for win in range(2 * max_window_size): file.write("NA,")
            file.write("%s" % wordpair[2])
            file.write("\n")


file1 = open("word_pair_similarity_WS353.csv", "w+", encoding="utf-8")

eval_wordpair_rank(file1, max_window_size, WS353File)

file1.close()

file2 = open("word_pair_similarity_SimLex.csv", "w+", encoding="utf-8")

eval_wordpair_rank(file2, max_window_size, SimLexFile)

file2.close()