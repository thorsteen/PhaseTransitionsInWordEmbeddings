# PhaseTransitionsInWordEmbeddings
Ba project by Thor Larsen at Institute of Computer Science @ UCPH supervised by Daniel Hershcovich

McKinney-Bock and Bedrick (2019) and Hershcovich et al. (2019) found anecdotally that word embedding of CBoW with context window size 1 judge similarity better than those trained with larger windows and performance improves gradually until it recovers. This bachelor thesis aims to research why this is happening, and what is changing in this phase transition. We reproduce Skip-gram and CBoW models with similar parameters on different context windows sizes and evaluate on the WS353 dataset. Furthermore, we look into relations between the most similar word pairs found by the word embedding and come up with hypothesis that semantically relations as sister terms can explain why model with narrow context windows perform better. We test our hypothesis on the SimLex dataset.

Models are created using Gensim's implementaion of the original Word2Vec model by Mikolov et.al., evaluated using Spearman's $\rho$ and our hypothesis is tested using a hyper-geometric significance test.

# Tentative schedule:
   • 6 april reading
   • 6 may reproduce word2vec exp.
   • 14 may write background: word2vec
   • 20 may benchmark qualitative analysis
   • 1 june: write background benchmarks
   • 10 june testing characteristics hypotheses
   • 1 july writing results
   • 20 july first draft
   • 14 august submission
