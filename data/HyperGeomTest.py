import numpy as np
from scipy.stats import hypergeom
import os
from itertools import product
import sys

IDENTITY = "identity"
SYNONYMS = "synonyms"
SYNSET_RELATIONS = ("hypernyms", "hyponyms") + \
                   tuple(map("_".join, product(("member", "part", "substance"), ("holonyms", "meronyms"))))
LEMMA_RELATIONS = "antonyms", "pertainyms", "derivationally_related_forms"
INDIRECT_PREFIX = "indirect_"
CO_PREFIX = "co_"
DIRECT_RELATIONS = (SYNONYMS, IDENTITY) + SYNSET_RELATIONS + LEMMA_RELATIONS


home = os.path.expanduser("~")

#os.chdir(home + "\Documents\BaProject\PhaseTransitionsInWordEmbeddings\Data\WordPairRelations")
os.chdir(home + "\Documents\BaProject\PhaseTransitionsInWordEmbeddings\Data\WordPairRelationsSimLex")

#os.remove("pvalues_relations_CBoW.csv")
#os.remove("Hypergeom_relation_test_counts.txt")


outfile = open("relation_count_pval.csv", "w")
outfile.write("Model")
for relation in DIRECT_RELATIONS: outfile.write(","+str(relation))
outfile.write("\n")


for win in range(25):
    infile = "CBoW_W{}_word_pair_relation.csv".format(win+1)
    outfile.write("W{},".format(win+1))
    data = np.loadtxt(infile, dtype = str, delimiter = ",")
    num_pairs = len(data)
    num_top = 90
    print("Running relation hypergeom-test of relations top-{} in ".format(num_top) + infile)
    print("----------------------------------------------")

    for relation in DIRECT_RELATIONS:
        relation_total_count = 0
        relation_top_count = 0
        print(relation)
        
        for i in range(num_pairs):
            if relation == data[i,2].strip() or INDIRECT_PREFIX + relation == data[i,2].strip() or CO_PREFIX + relation == data[i,2].strip():
                relation_total_count += 1
                if num_top > i:
                    relation_top_count += 1
        #use hyper-geometric survival function to calc p-values
        #pval = hypergeom.sf(x-1, M, n, N)
        outfile.write("{} ({:.2f}),".format(relation_top_count, hypergeom.sf(relation_top_count - 1, num_pairs, relation_total_count, num_top)))
        print("relation top count: "+str(relation_top_count))
        print("relation total count: "+str(relation_total_count))
        print("p-value: "+str(hypergeom.sf(relation_top_count - 1, num_pairs, relation_total_count, num_top)))
        print("")
    outfile.write("\n")
        
outfile.close()
