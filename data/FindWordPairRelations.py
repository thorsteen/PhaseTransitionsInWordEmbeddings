import os
from itertools import product, chain
import pandas as pd
from nltk.corpus import wordnet as wn

home = os.path.expanduser("~")
os.chdir(home + "\Documents\BaProject\PhaseTransitionsInWordEmbeddings\Data")


results = pd.read_excel("model_word_pair_comparison_ws353.xlsx")
#remove missing
results.drop(results[results.cbow_model_w1 == 'missing'].index,inplace = True)
#reduced to 85 word pairs

results_SimLex = pd.read_csv("word_pair_similarity_SimLex.csv")
results_SimLex.dropna(inplace=True)
#reduced to 267 wordpairs

IDENTITY = "identity"
SYNONYMS = "synonyms"
SYNSET_RELATIONS = ("hypernyms", "hyponyms") + \
                   tuple(map("_".join, product(("member", "part", "substance"), ("holonyms", "meronyms"))))
OTHER_SYNSET_RELATIONS = ("entailments", "instance_hypernyms", "instance_hyponyms",
                          "topic_domains", "region_domains", "usage_domains", "attributes", "causes", "also_sees",
                          "verb_groups", "similar_tos")
LEMMA_RELATIONS = "antonyms", "pertainyms", "derivationally_related_forms"
INDIRECT_PREFIX = "indirect_"
CO_PREFIX = "co_"
DIRECT_RELATIONS = (SYNONYMS,) + SYNSET_RELATIONS + OTHER_SYNSET_RELATIONS + LEMMA_RELATIONS
UNIQUE_RELATIONS = ("identity", "synonyms")

DEPTH = 20


def gen_related(word, relations=None):
    lemmas = wn.lemmas(word)
    if not relations or IDENTITY in relations:
        yield IDENTITY, lemmas
    synsets = wn.synsets(word)
    synonyms = {lemma for synset in synsets for lemma in synset.lemmas()}
    if not relations or SYNONYMS in relations:
        yield SYNONYMS, synonyms
    related_synsets = {}
    related_lemmas = {}

    def _r(s):
        return getattr(s, relation)()
    for relation in SYNSET_RELATIONS + OTHER_SYNSET_RELATIONS:
        if not relations or relation in relations:
            related_synsets[relation] = {synset for syn in synsets for synset in _r(syn)}
            related_lemmas[relation] = {lemma for synset in related_synsets[relation] for lemma in synset.lemmas()}
            yield relation, related_lemmas[relation]
    for relation in SYNSET_RELATIONS:
        if not relations or INDIRECT_PREFIX + relation in relations:
            yield INDIRECT_PREFIX + relation, {lemma for syn in related_synsets[relation]
                                               for synset in syn.closure(_r, DEPTH)
                                               if synset not in related_synsets[relation]
                                               for lemma in synset.lemmas() if lemma not in related_lemmas[relation]}
    for relation in LEMMA_RELATIONS:
        if not relations or relation in relations:
            yield relation, {lemma for lemma1 in lemmas for lemma in _r(lemma1)}
    for relation, inverse in zip(SYNSET_RELATIONS[::2], SYNSET_RELATIONS[1::2]):
        if not relations or CO_PREFIX + relation in relations:
            related_synsets_co = {synset2 for synset1 in related_synsets[inverse] for synset2 in _r(synset1)}
            yield CO_PREFIX + relation, {lemma for synset2 in related_synsets_co for lemma in synset2.lemmas()}


def find_relations(words, by_pair=False):
    pivot, *candidates = words
    candidate_lemmas = [set(wn.lemmas(w)) for w in candidates]
    relations = {}
    for relation, related in gen_related(pivot):
        related_lemmas = List(related)
        for candidate, lemmas in zip(candidates, candidate_lemmas):
            if (pivot, candidate) not in chain(*(relations.get(r, ()) for r in UNIQUE_RELATIONS)) and \
                    any(lemma in lemmas for lemma in related_lemmas):
                relations.setdefault(relation, []).append((pivot, candidate))
    if by_pair:
        pair_relations = {(pivot, c): [] for c in candidates} if by_pair == "all" else {}
        for relation, pairs in relations.items():
            for pair in pairs:
                pair_relations.setdefault(pair, []).append(relation)
        return pair_relations
    else:
        return relations


def all_relations(lists):
    relation_pairs = {}
    for words in lists:
        for relation, pairs in find_relations(words).items():
            relation_pairs.setdefault(relation, []).extend(pairs)
    return relation_pairs

def check_relations(word1, word2):
        relation_pairs = gen_related(word1)

        for relation_name, lemmas in relation_pairs:
            for synset in wn.synsets(word2):
                for lemma in synset.lemmas():
                    if lemma in lemmas:
                        return relation_name


#we want to compare to ws353 ranking
ws_353_scores = results.sort_values("WS_similiarity_score",ascending=False, ignore_index=True)
    
for win in range(25):
    #sort in decedning order for first sim score (cosine sim)
    results.sort_values('cbow_model_w{}'.format(win+1), ascending=False, ignore_index=True, inplace = True)
    #write results from wordnet to text file
    
    f = open('WordPairRelations/CBoW_W{}_word_pair_relation.csv'.format(win+1), 'w+')
    
    #f.write("Word1,Word2,Relation\n")
    
    for i in range(len(results)):
        f.write("{},{},{} \n".format(results.iloc[i,0],results.iloc[i,1],check_relations(results.iloc[i,0],results.iloc[i,1])))
        
    f.close()

#we want to do the same with SimLex
SimLex_scores = results_SimLex.sort_values("Reference_val ",ascending=False, ignore_index=True)
    
for win in range(25):
    #sort in decedning order for first sim score (cosine sim)
    results_SimLex.sort_values('CBoW_Model_W{}'.format(win+1), ascending=False, ignore_index=True, inplace = True)
    #write results from wordnet to text file
    
    f = open('WordPairRelationsSimLex/CBoW_W{}_word_pair_relation.csv'.format(win+1), 'w+')
    
    #f.write("Word1,Word2,Relation\n")
    
    for i in range(len(results)):
        f.write("{},{},{} \n".format(results.iloc[i,0],results.iloc[i,1],check_relations(results.iloc[i,0],results.iloc[i,1])))
        
    f.close()     
