# -*- coding: utf-8 -*-

__author__ = 'Jasneet Sabharwal <jsabharw@sfu.ca>'

import optparse
import sys
import os
import time
from collections import defaultdict

from src.tools.fileReaders import readPartialAlignments
from src.models.IBMModel1 import supervisedIBMModel1, interpolatedIBMModel1
from src.unsupervised.featurized_hmm_mp_e_step_parallel_theta_efficient import get_features_fired


def runIBMModel1(sVocabCount, tVocabCount, stCoOccurrenceCount, bitext, partialAlignments, interpolationWeight):
    """
    Execute supervised IBM Model 1 and then execute unsupervised-iterpolated IBM Model1
    :param sVocabCount: source vocab occurrence count
    :param tVocabCount: target vocab occurrence count
    :param stCoOccurrenceCount: source and target word co-occurrence count
    :param bitext: parallel source and target sentences
    :param partialAlignments: partial alignments
    :param interpolationWeight: lambda for linear interpolation of two models
    :return: translation probability, t(f|e)
    """
    startTime = time.time()

    tProb = supervisedIBMModel1(stCoOccurrenceCount, bitext, partialAlignments)
    tProb = interpolatedIBMModel1(sVocabCount, tVocabCount, stCoOccurrenceCount, bitext, tProb, interpolationWeight)

    endTime = time.time()
    print "RUN TIME FOR IBM MODEL! %.2gs" % (endTime - startTime)

    return tProb

if __name__ == '__main__':
    optParser = optparse.OptionParser()
    optParser.add_option("-p", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
    optParser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
    optParser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
    optParser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
    optParser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
    optParser.add_option("-a", "--annotation", dest="annotation", defaul=None, help="Partial annotations of source and target")
    optParser.add_option("-asc", "--annotationSourceColumn", dest="annotationSourceColumn", defaul=0, type="int", help="Partial annotations column number of source words in file")
    optParser.add_option("-atc", "--annotationTargetColumn", dest="annotationTargetColumn", defaul=1, type="int", help="Partial annotations Column number of target words in file")
    optParser.add_option("-l1", "--ibm1Lambda", dest="ibm1Lambda", default=0.5, type="float", help="Interpolation lambda for IBM Model 1 (default=0.5)")

    (opts, _) = optParser.parse_args()

    fData = "%s.%s" % (opts.train, opts.french)
    eData = "%s.%s" % (opts.train, opts.english)
    testFData = fData
    testEData = eData

    # Check if fData and eData files exist
    if not (os.path.isfile(fData) and os.path.isfile(eData)):
        print >>sys.stderr, __doc__.strip('\n\r')
        sys.exit(1)

    sys.stderr.write('#'*10 + ' BEGINNING TRAINING ' + '#'*10)

    # Read data from fData and eData and store them as bitextFE and bitextEF
    with open(fData) as fFile, open(eData) as eFile:
        bitextFE = [[sentence.strip().split() for sentence in pair] for pair in zip(fData, eData)[:opts.num_sents]]
        bitextEF = [[sentence.strip().split() for sentence in pair] for pair in zip(eFile, fFile)[:opts.num_sents]]

    # Read testdata bitext
    with open(testFData) as fFile, open(testEData) as eFile:
        bitextTest = [[sentence.strip().split() for sentence in pair] for pair in zip(fFile, eFile)[:opts.num_sents]]

    # Initialize variables (count variables for gathering vocab counts, co-occurrence counts)
    fCount = defaultdict(int)
    eCount = defaultdict(int)
    feCount = defaultdict(int)
    efCount = defaultdict(int)
    normalizingDecisionMap = defaultdict(list)
    featureIndex = defaultdict(int)
    fVector = defaultdict(int)

    # For multiprocessing
    eventIndex = set([])

    for (n, (f, e)) in enumerate(bitextFE):
        for f_i in set(f):
            fCount[f_i] += 1
            for e_j in set(e):
                feCount[(f_i, e_j)] += 1
                efCount[(e_j, f_i)] += 1

                eventIndex.add((f_i, e_j))

                if f_i not in normalizingDecisionMap[e_j]:
                    normalizingDecisionMap[e_j].append(f_i)

                featuresList = get_features_fired(f_i, e_j)
                for feature in featuresList:
                    if feature not in featureIndex:
                        featureIndex[feature] = len(featureIndex)
                    fVector[featureIndex[feature]] += 1
        for e_j in e: # TODO: Confirm if it should be set of e or not like set of f above
            eCount[e_j] += 1

    eventIndex = sorted(list(eventIndex))

    # Run IBM Model 1
    partialAlignments = readPartialAlignments(opts.annotation, opts.annotationSourceColumn, opts.annotationTargetColumn)
    runIBMModel1(fCount, eCount, feCount, bitextFE, partialAlignments, opts.ibm1Lambda)
