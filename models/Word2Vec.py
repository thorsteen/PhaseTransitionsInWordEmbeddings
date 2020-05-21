# import modules & set up logging
import gensim, logging
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = gensim.models.word2vec.Text8Corpus("../Data/text8")

similarities = []

for i in range(1, 26):
    
    model = gensim.models.Word2Vec(sentences, min_count=500, size=500, workers=6, window = i+1, sg = 0, cbow_mean = 0) 
    #Training algorithm: 1 for skip-gram; otherwise CBOW.
    #cbow_mean ({0, 1}, optional) – If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
    word_vectors = model.wv
    del model
    similarities.append(word_vectors.evaluate_word_pairs('../Data/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt'))
    del word_vectors
    
print(similarities)

rho_CBOW = []

for i in range(len(similarities)):
    rho_CBOW.append(similarities[i][1])

window_size_CBOW = np.arange(1,len(rho_CBOW)+1,1)
print("CBOW evaluation on WS-353 Spearmans rho values, windows size 1 - {}".format(len(rho_CBOW)))
CBOW_plot, ax_CBOW = plt.subplots()
ax_CBOW.plot(window_size_CBOW, rho_CBOW)
plt.xticks(window_size_CBOW)
ax_CBOW.set_title("CBOW evaluation WS-353 Spearmans rho values, windows size 1 - {}".format(len(rho_CBOW)))
CBOW_plot.savefig("WS-353_Spearmans_rho_CBOW_plot.png")


#storing of models
#model.save('/tmp/mymodel')
#new_model = gensim.models.Word2Vec.load('/tmp/mymodel')

#model = gensim.models.Word2Vec.load('/tmp/mymodel')
#model.train(more_sentences)

#for similiarity score
#model.most_similar
#model.doesnt_match
#model.similarity

#to get vectors of a words
#model['word']

#also vecotrs in numpy array
#model.syn0
