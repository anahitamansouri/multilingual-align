#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'amansour'

#!/usr/bin/env python

"""
    python align.py [options]

    options are:

    -p DIR/PREFIX   prefix for parallel data, Defaults: DIR=../ PREFIX=../hansards when running from answer directory
    -n NUMBER       number of training examples to use, Default: n = sys.maxint
"""
import optparse
import os.path

import sys

from IBM_Model1 import EM_IBM1


#from HMM import baumWelch, findBestAlignmentsForAll, findBestAlignmentsForAllWithIntersection, findBestAlignmentsForAll_AER, findBestAlignmentsForAllWithIntersection_AER, baumWelchP
from src.unsupervised.HMM_with_length_with_array_null import findBestAlignmentsForAll_AER, baumWelchP
from src.unsupervised.featurized_hmm_mp_e_step_parallel_theta_efficient import get_features_fired, em_with_features
from evaluate import grade_align, convert_giza_out_to_aer_out
from collections import defaultdict

import time


optparser = optparse.OptionParser()
optparser.add_option("-p", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)
#i_data = "project/hansards.it"
#es_data = "project/hansards.es"

test_f_data = f_data

test_e_data = e_data

#alignment = 'alignment'

if not ( os.path.isfile(f_data) and os.path.isfile(e_data) ):
    print >>sys.stderr, __doc__.strip('\n\r')
    sys.exit(1)

sys.stderr.write("Training started...")

bitext_fe = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
bitext_ef = [[sentence.strip().split() for sentence in pair] for pair in zip(open(e_data), open(f_data))[:opts.num_sents]]


bitext_test = [[sentence.strip().split() for sentence in pair] for pair in zip(open(test_f_data), open(test_e_data))[:opts.num_sents]]

f_count = defaultdict(int)
e_count = defaultdict(int)

fe_count = defaultdict(int)
ef_count = defaultdict(int)

normalizing_decision_map = defaultdict(list)

feature_index = defaultdict(int)

f_vector = defaultdict(int)

#Useful for multiprocessing
event_index = set([])

for (n, (f, e)) in enumerate(bitext_fe):
  for f_i in set(f):
    f_count[f_i] += 1
    for e_j in set(e):
      fe_count[(f_i,e_j)] += 1
      ef_count[(e_j,f_i)] += 1

      event_index.add((f_i, e_j))

      if f_i not in normalizing_decision_map[e_j]:
          normalizing_decision_map[e_j] += [f_i]

      features_list = get_features_fired(f_i,e_j)
      for feature in features_list:
          if feature not in feature_index:
              feature_index[feature] = len(feature_index)
          f_vector[feature_index[feature]] += 1
  for e_j in e:
    e_count[e_j] += 1

event_index = sorted(list(event_index))
#print 'normal ', len(normalizing_decision_map['the'])
#print 'f vec ', f_vector[feature_index[('EMISSION','le','the')]]
def print_alignment_SD_ibm1(bitext,t_sd, alignmentFile, num_lines):
    alignment = open('alignment','w')
    for (n,(S, D)) in enumerate(bitext):
        set1 = set()
        #set2 = set()
        i = 0
        for s_i in S:
            max_p = 0
            argmax = -1
            j = 0
            for d in D:
                if t_sd[(s_i,d)] > max_p:
                    max_p = t_sd[(s_i,d)]
                    argmax = j
                j += 1
            set1.add((i,argmax))
            i += 1
        for (i_s,argmax_s) in set1:
            sys.stdout.write("%i-%i " % (i_s+1,argmax_s+1))
            alignment.write("%i-%i " % (i_s+1,argmax_s+1))
	sys.stdout.write("\n")
	alignment.write("\n")
        if n == num_lines-1:
                return
    return alignment

def intersect_SD_with_DS_Models(bitext,p_sd,p_ds):
    for (n,(S, D)) in enumerate(bitext):
        set1 = set()
        set2 = set()
        i = 0
        for s_i in S:
            max_p = 0
            argmax = -1
            j = 0
            for d in D:
                if p_sd[(s_i,d)] > max_p:
                    max_p = p_sd[(s_i,d)]
                    argmax = j
                j += 1
            set1.add((i,argmax))
            i += 1
        i = 0
        for d_i in D:
            max_p = 0
            argmax = -1
            j = 0
            for s in S:
                if p_ds[(d_i,s)] > max_p:
                    max_p = p_ds[(d_i,s)]
                    argmax = j
                j += 1
            set2.add((argmax,i))
            i += 1
            intersect = set1.intersection(set2)
        for (i_s,argmax_s) in intersect:
            sys.stdout.write("%i-%i " % (i_s,argmax_s))
        sys.stdout.write("\n")


def intersect_SD_with_DS_Models(bitext,p_sd,p_ds,num_lines):
    alignment = open('alignment','w')
    for (n,(S, D)) in enumerate(bitext):
        set1 = set()
        set2 = set()
        i = 1
        for s_i in S:
            max_p = 0
            argmax = -1
            j = 1
            for d in D:
                if p_sd[(s_i,d)] > max_p:
                    max_p = p_sd[(s_i,d)]
                    argmax = j
                j += 1
            set1.add((i,argmax))
            i += 1
        i = 1
        for d_i in D:
            max_p = 0
            argmax = -1
            j = 1
            for s in S:
                if p_ds[(d_i,s)] > max_p:
                    max_p = p_ds[(d_i,s)]
                    argmax = j
                j += 1
            set2.add((argmax,i))
            i += 1
            intersect = set1.intersection(set2)
        for (i_s,argmax_s) in intersect:
            alignment.write("%i-%i " % (i_s,argmax_s))
        #alignment.append(intersect)
        alignment.write("\n")
        if n == num_lines-1:
                return
    return alignment

def read_de_en_file(file_name):
    file1 = open(file_name, 'r')
    de_file = open('train.de', 'w')
    en_file = open('train.en', 'w')
    for line in file1:
        [de_sentence,en_sentence] = line.split(' ||| ')
        de_file.write(de_sentence)
        de_file.write("\n")
        en_file.write(en_sentence)
        #en_file.write("\n")
    de_file.close()
    en_file.close()

def run_IBM1():
    t_fe = EM_IBM1(f_count,e_count,fe_count,bitext_fe)
    return t_fe

def run_HMM(t_fe):
    startTime = time.time()
    (a,b,pi) = baumWelchP(bitext_fe, f_count, t_fe,fe_count)
    #(a,b,pi) = em_with_features(bitext_fe, f_count, t_fe,fe_count)
    endTime = time.time()
    print "run time for hmm model %.2gs" % (endTime - startTime)
    return (a,b,pi)

def run_HMM_with_length(t_fe):
    startTime = time.time()
    (a,b,pi) = baumWelchP(bitext_fe, f_count, t_fe,fe_count)
    #(a,b,pi) = em_with_features(bitext_fe, f_count, t_fe,fe_count)
    endTime = time.time()
    print "run time for hmm model %.2gs" % (endTime - startTime)
    return (a,b,pi)

def run_HMM_Null(t_fe):
    startTime = time.time()
    (a,b,pi) = baumWelchP(bitext_fe, f_count, t_fe,fe_count)
    #(a,b,pi) = em_with_features(bitext_fe, f_count, t_fe,fe_count)
    endTime = time.time()
    print "run time for hmm model %.2gs" % (endTime - startTime)
    return (a,b,pi)

def run_featurized_HMM():
    print 'kappa',kappa
    startTime = time.time()
    (a, b, pi) = em_with_features(bitext_fe, f_count, t_fe, fe_count, feature_index, normalizing_decision_map, kappa, f_vector, event_index)
    endTime = time.time()
    print "run time for featurized model %.2gs" % (endTime - startTime)
    return (a, b, pi)
def run_giza():
    global alignment
    alignment = 'alignment_de-en_aer_format'
    convert_giza_out_to_aer_out('aligned.srctotgt.de-en',alignment)

#t_fe = EM_IBM1(f_count,e_count,fe_count,bitext_fe)
#t_ef = EM_IBM1(e_count,f_count,ef_count,bitext_ef)
#print len(t_fe), t_fe[('le','the')]
#print_alignment_SD_ibm1(bitext_test,t_fe, alignment)
#print intersect_SD_with_DS_Models(bitext_test,t_fe,t_ef)
#intersect_SD_with_DS_Models(bitext_test,t_fe,t_ef,100)

#(a,b,pi) = baumWelch(bitext_fe, f_count, t_fe,fe_count)
#(a_ef,b_ef,pi_ef) = baumWelch(bitext_ef, e_count, t_ef,ef_count)

#print 'len feature index ', len(feature_index), ' ', len(t_fe), ' ', len(f_vector)
kappa = 0.0

#get_gradient_with_counts(ones(len(feature_index)),e,kappa,feature_index,normalizing_decision_map,f_vector)
#get_likelihood_with_expected_counts(ones(len(feature_index)), e, kappa,feature_index,normalizing_decision_map, f_vector)
#sys.exit(1)
'''startTime = time.time()
(a, b, pi) = em_with_features(bitext_fe, f_count, t_fe, fe_count, feature_index, normalizing_decision_map, kappa, f_vector, event_index)
#(a_ef, b_ef, pi_ef) = em_with_features(bitext_ef, e_count, t_ef, ef_count, feature_index_ef, normalizing_decision_map_ef, kappa, f_vector_ef, event_index_ef)
endTime = time.time()
print "run time for featurized model %.2gs" % (endTime - startTime)
#(a_ef,b_ef,pi_ef) = em_with_features(bitext_ef, e_count, t_ef, ef_count, feature_index, normalizing_decision_map, kappa=1)
'''
alignment = "alignment"
#gold for french-english
gold = "/cs/natlang-user/amansour/test.aligned.txt"
#gold for german-english
#gold = "/cs/natlang-user/amansour/dev"
output = sys.stdout
#run_giza()
#gold =  "/cs/natlang-user/amansour/de-en.test.aligned.txt"

#convert_giza_out_to_aer_out('../dev.align',gold)
#read_de_en_file('../dev-test-train.de-en')

#print_alignment_SD_ibm1(bitext_test,t_fe, alignment, 100)
kappa = 0.0
t_fe = run_IBM1()
(a, b, pi) = run_HMM_with_length(t_fe)
#(a, b, pi) = run_HMM_Null(t_fe)
#(a, b, pi) = run_featurized_HMM()

#findBestAlignmentsForAll(bitext_fe,a,b,pi)
#findBestAlignmentsForAllWithIntersection_DS(bitext_fe,a,b,pi,a_ef,b_ef,pi_ef)
#findBestAlignmentsForAll(bitext_ef,a_ef,b_ef,pi_ef)
#findBestAlignmentsForAllWithIntersection(bitext_test, a, b, pi, a_ef, b_ef, pi_ef)
findBestAlignmentsForAll_AER(bitext_test,a,b,pi,100, alignment)
#findBestAlignmentsForAllWithIntersection_AER(bitext_test, a, b, pi, a_ef, b_ef, pi_ef,448,alignment)

grade_align(f_data, e_data, gold, alignment,output)

#findBestAlignmentsForAll_AER(bitext_test,a,b,pi,100, alignment)
#grade_align(f_data, e_data, gold, "alignment_5000_BT_100_mist_HMM",output)
