# import modules & set up logging
import gensim, logging
import numpy as np
import matplotlib.pyplot as plt
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

os.chdir(os.getcwd()+"\Documents\BaProject\PhaseTransitionsInWordEmbeddings\models")

sentences = gensim.models.word2vec.Text8Corpus("../Data/text8")

max_window_size = 25

cbow_similarities = []

for i in range(1, max_window_size+1):
    
    model = gensim.models.Word2Vec(sentences, min_count=500, size=500, workers=6, window = i, sg = 0, cbow_mean = 0) 
    #Training algorithm: 1 for skip-gram; otherwise CBOW.
    #cbow_mean ({0, 1}, optional) – If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
    word_vectors = model.wv
    model.save('Gensim_word2vec_models/cbow_model_w{}'.format(i))
    del model
    cbow_similarities.append(word_vectors.evaluate_word_pairs('../Data/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt'))
    del word_vectors

print("CBOW evaluation on WS-353 Spearmans rho values, windows size 1 - {}".format(len(cbow_similarities)))
print(cbow_similarities)

rho_CBOW = []

for i in range(len(cbow_similarities)):
    rho_CBOW.append(cbow_similarities[i][1])

window_size_CBOW = np.arange(1,len(rho_CBOW)+1,1)
CBOW_plot, ax_CBOW = plt.subplots()
ax_CBOW.spines["top"].set_visible(False)    
ax_CBOW.spines["bottom"].set_visible(False)    
ax_CBOW.spines["right"].set_visible(False)    
ax_CBOW.spines["left"].set_visible(False)  
ax_CBOW.plot(window_size_CBOW, rho_CBOW, color='black')
plt.xticks(window_size_CBOW)
plt.ylim(0.4, 0.8)
plt.xlim(1, len(window_size_CBOW)) 
ax_CBOW.set_title("CBoW evaluation on WS-353")
plt.ylabel("Spearman's rho values")
plt.xlabel("window size 1 - {}".format(len(rho_CBOW)))
CBOW_plot.set_size_inches(11.69,4)
CBOW_plot.savefig("WS-353_Spearmans_rho_CBOW_plot.png")


SGNS_similarities = []

for i in range(1, max_window_size+1):
    
    model = gensim.models.Word2Vec(sentences, min_count=500, size=500, workers=6, window = i, sg = 1) 
    #Training algorithm: 1 for skip-gram; otherwise CBOW.
    #cbow_mean ({0, 1}, optional) – If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
    word_vectors = model.wv
    model.save('Gensim_word2vec_models/SGNS_model_w{}'.format(i))
    del model
    SGNS_similarities.append(word_vectors.evaluate_word_pairs('../Data/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt'))
    del word_vectors


print("Skip-gram evaluation on WS-353 Spearmans rho values, windows size 1 - {}".format(len(SGNS_similarities)))
print(SGNS_similarities)

rho_SGNS = []

for i in range(len(SGNS_similarities)):
    rho_SGNS.append(SGNS_similarities[i][1])

window_size_SGNS = np.arange(1,len(rho_SGNS)+1,1)
SGNS_plot, ax_SGNS = plt.subplots()
ax_SGNS.spines["top"].set_visible(False)    
ax_SGNS.spines["bottom"].set_visible(False)    
ax_SGNS.spines["right"].set_visible(False)    
ax_SGNS.spines["left"].set_visible(False)  
ax_SGNS.plot(window_size_SGNS, rho_SGNS, color='black')
plt.xticks(window_size_SGNS)
plt.ylim(0.4, 0.8)
plt.xlim(1, len(window_size_SGNS))  
ax_SGNS.set_title("Skip-gram evaluation on WS-353")
plt.ylabel("Spearman's rho values")
plt.xlabel("window size 1 - {}".format(len(rho_SGNS)))
SGNS_plot.set_size_inches(11.69,4)
SGNS_plot.savefig("WS-353_Spearmans_rho_SGNS_plot.png")


os.chdir(os.getcwd()+"\Gensim_word2vec_models")

virus_cbow = []
virus_sgns = []


file = writtenByFile = open("virus_word_pair_comparison.txt", "w+", encoding="utf-8")

for win in range(max_window_size):
    virus_cbow.append(('cbow_model_w'+str(win+1),gensim.models.Word2Vec.load('cbow_model_w'+str(win+1)).wv.most_similar("virus")))
    virus_sgns.append(('SGNS_model_w'+str(win+1),gensim.models.Word2Vec.load('SGNS_model_w'+str(win+1)).wv.most_similar("virus")))
    file.write("%s\n" % (str(virus_cbow[win])))
    file.write("%s\n" % (str(virus_sgns[win])))

file.close()
#for similiarity score
#model.most_similar('word')
#model.doesnt_match('word')
#model.similarity('word')

#to get vectors of a words
#model['word']
