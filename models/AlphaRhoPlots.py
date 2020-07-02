# import modules & set up logging
import gensim, logging
import numpy as np
import matplotlib.pyplot as plt
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

home = os.path.expanduser("~")
os.chdir(home+"\Documents\BaProject\PhaseTransitionsInWordEmbeddings\models")


max_window_size = 25

cbow_similarities_WS = []
cbow_similarities_Sim = []


#preprocessing of simlex
def pre_proces_simlex(filename):
    sim999 = np.loadtxt(filename,dtype=str)
    
    sim999 = np.array((sim999[:,0],sim999[:,1],sim999[:,3])).transpose()
    
    a_file = open("SimLex-999-1.txt", "w+")
    np.savetxt(a_file, sim999, encoding = "utf-8", delimiter = " ", fmt='%s')
    a_file.close()
    
#pre_proces_simlex('../Data/SimLex-999/SimLex-999.txt')

for i in range(1, max_window_size+1):
    
    #model = gensim.models.Word2Vec(sentences, min_count=500, size=500, workers=6, window = i, sg = 0, cbow_mean = 0) 
    #Training algorithm: 1 for skip-gram; otherwise CBOW.
    #cbow_mean ({0, 1}, optional) – If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
    
    model = gensim.models.Word2Vec.load('Gensim_word2vec_models/cbow_model_w{}'.format(i))
    word_vectors = model.wv
    del model
    cbow_similarities_WS.append(word_vectors.evaluate_word_pairs('../Data/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt'))
    cbow_similarities_Sim.append(word_vectors.evaluate_word_pairs('../Data/SimLex-999/SimLex-999-1.txt',delimiter = " "))
    del word_vectors

print("CBOW evaluation on WS-353 Spearmans rho values, windows size 1 - {}".format(len(cbow_similarities_WS)))
print("WS353")
print(cbow_similarities_WS)
print("SimLex")
print(cbow_similarities_Sim)


rho_CBOW = []

for i in range(len(cbow_similarities_Sim)):
    rho_CBOW.append(cbow_similarities_Sim[i][1])

window_size_CBOW = np.arange(1,len(rho_CBOW)+1,1)
CBOW_plot, ax_CBOW = plt.subplots()
ax_CBOW.spines["top"].set_visible(False)    
ax_CBOW.spines["bottom"].set_visible(False)    
ax_CBOW.spines["right"].set_visible(False)    
ax_CBOW.spines["left"].set_visible(False)  
ax_CBOW.plot(window_size_CBOW, rho_CBOW, color='black')
plt.xticks(window_size_CBOW)
plt.ylim(0.1, 0.5)
plt.xlim(1, len(window_size_CBOW)) 
ax_CBOW.set_title("CBoW evaluation on SimLex")
plt.ylabel("Spearman's rho values")
plt.xlabel("window size 1 - {}".format(len(rho_CBOW)))
CBOW_plot.set_size_inches(11.69,4)
CBOW_plot.savefig("SimLex_Spearmans_rho_CBOW_plot.png")


SGNS_similarities_WS = []
SGNS_similarities_Sim = []


for i in range(1, max_window_size+1):
    
    #model = gensim.models.Word2Vec(sentences, min_count=500, size=500, workers=6, window = i, sg = 1) 
    #Training algorithm: 1 for skip-gram; otherwise CBOW.
    #cbow_mean ({0, 1}, optional) – If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
    #model.save('Gensim_word2vec_models/SGNS_model_w{}'.format(i))
    model = gensim.models.Word2Vec.load('Gensim_word2vec_models/SGNS_model_w{}'.format(i))
    word_vectors = model.wv

    del model
    SGNS_similarities_WS.append(word_vectors.evaluate_word_pairs('../Data/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt'))
    SGNS_similarities_Sim.append(word_vectors.evaluate_word_pairs('../Data/SimLex-999/SimLex-999-1.txt', delimiter = " "))

    del word_vectors


print("Skip-gram evaluation on WS-353 Spearmans rho values, windows size 1 - {}".format(len(SGNS_similarities_WS)))
print(SGNS_similarities_WS)

rho_SGNS = []

for i in range(len(SGNS_similarities_Sim)):
    rho_SGNS.append(SGNS_similarities_Sim[i][1])

window_size_SGNS = np.arange(1,len(rho_SGNS)+1,1)
SGNS_plot, ax_SGNS = plt.subplots()
ax_SGNS.spines["top"].set_visible(False)    
ax_SGNS.spines["bottom"].set_visible(False)    
ax_SGNS.spines["right"].set_visible(False)    
ax_SGNS.spines["left"].set_visible(False)  
ax_SGNS.plot(window_size_SGNS, rho_SGNS, color='black')
plt.xticks(window_size_SGNS)
plt.ylim(0.1, 0.5)
plt.xlim(1, len(window_size_SGNS))  
ax_SGNS.set_title("Skip-gram evaluation on SimLex")
plt.ylabel("Spearman's rho values")
plt.xlabel("window size 1 - {}".format(len(rho_SGNS)))
SGNS_plot.set_size_inches(11.69,4)
SGNS_plot.savefig("Sim_Spearmans_rho_SGNS_plot.png")
