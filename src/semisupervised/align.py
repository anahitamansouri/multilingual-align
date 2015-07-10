# -*- coding: utf-8 -*-

__author__ = 'Jasneet Sabharwal <jsabharw@sfu.ca>'

import argparse
import sys
import os
import time
from collections import defaultdict

from src.tools.fileReaders import readPartialAlignments
from src.models.IBMModel1 import supervisedIBMModel1, interpolatedIBMModel1, unsupervisedIBMModel1
from src.models.IBMModel2 import supervisedIBMModel2, interpolatedIBMModel2, unsupervisedIBMModel2
from src.unsupervised.featurized_hmm_mp_e_step_parallel_theta_efficient import get_features_fired
from src.models.HMM import supervisedHMM
from src.models.HMM_with_length_with_array_null import interpolatedBaumWelchP, findBestAlignmentsForAll_AER
from src.unsupervised.HMM_with_length_with_array_null import baumWelchP
from src.tools.evaluate import grade_align
from src.tools.alignment import alignmentFromIBM1, alignmentFromIBM2

def runIBMModel1(stCoOccurrenceCount, bitext, partialAlignments, interpolationWeight,
                 interpolate=True):
    """
    Execute supervised IBM Model 1 and then execute unsupervised-iterpolated IBM Model1
    :param sVocabCount: source vocab occurrence count
    :param tVocabCount: target vocab occurrence count
    :param stCoOccurrenceCount: source and target word co-occurrence count
    :param bitext: parallel source and target sentences
    :param partialAlignments: partial alignments
    :param interpolationWeight: lambda for linear interpolation of two models
    :param interpolate: If interpolation should be performed or should use the unsupervised IBM Model1
    :return: translation probability, t(f|e)
    """
    startTime = time.time()

    if interpolate:
        tProb, qProb = supervisedIBMModel1(stCoOccurrenceCount, bitext, partialAlignments)
        tProb, qProb = interpolatedIBMModel1(stCoOccurrenceCount, bitext, tProb, interpolationWeight)
    else:
        tProb, qProb = unsupervisedIBMModel1(stCoOccurrenceCount, bitext)

    endTime = time.time()
    print "RUN TIME FOR IBM MODEL 1! %.2gs" % (endTime - startTime)

    return tProb, qProb


def runIBMModel2(stCoOccurrenceCount, bitext, ibm1TProb, ibm1QProb, partialAlignments, interpolationWeight,
                 interpolate=True):
    """
    Execute supervised IBM Model 1 and then execute unsupervised-iterpolated IBM Model1
    :param sVocabCount: source vocab occurrence count
    :param tVocabCount: target vocab occurrence count
    :param stCoOccurrenceCount: source and target word co-occurrence count
    :param bitext: parallel source and target sentences
    :param partialAlignments: partial alignments
    :param interpolationWeight: lambda for linear interpolation of two models
    :param interpolate: If interpolation should be performed or should use the unsupervised IBM Model1
    :return: translation probability, t(f|e)
    """
    startTime = time.time()

    if interpolate:
        tProb, qProb = supervisedIBMModel2(stCoOccurrenceCount, bitext, partialAlignments)
        tProb, qProb = interpolatedIBMModel2(stCoOccurrenceCount, bitext, ibm1TProb, ibm1QProb, tProb, qProb,
                                             interpolationWeight)
    else:
        tProb, qProb = unsupervisedIBMModel2(stCoOccurrenceCount, bitext, ibm1TProb, ibm1QProb)

    endTime = time.time()
    print "RUN TIME FOR IBM MODEL 2! %.2gs" % (endTime - startTime)

    return tProb, qProb


def runHMM(sourceVocabCount, stCoOccurrenceCount, bitext, ibmTProb, partialAlignments, interpolationWeight,
           interpolate=True):
    startTime = time.time()

    if interpolate:
        transitionMatrix, emissionMatrix = supervisedHMM(stCoOccurrenceCount, bitext, partialAlignments)

        transitionMatrix, emissionMatrix, stateDistribution = interpolatedBaumWelchP(bitext, sourceVocabCount, ibmTProb,
                                                                                     stCoOccurrenceCount,
                                                                                     transitionMatrix,
                                                                                     emissionMatrix,
                                                                                     interpolationWeight)
    else:
        transitionMatrix, emissionMatrix, stateDistribution = baumWelchP(bitext, sourceVocabCount, ibmTProb,
                                                                         stCoOccurrenceCount)

    endTime = time.time()
    print "RUN TIME FOR HMM! %.2gs" % (endTime - startTime)

    return transitionMatrix, emissionMatrix, stateDistribution


if __name__ == '__main__':
    optParser = argparse.ArgumentParser()
    optParser.add_argument("-p", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
    optParser.add_argument("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
    optParser.add_argument("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
    optParser.add_argument("-t", "--threshold", dest="threshold", default=0.5, type=float, help="Threshold for aligning with Dice's coefficient (default=0.5)")
    optParser.add_argument("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type=int, help="Number of sentences to use for training and alignment")
    optParser.add_argument("-a", "--annotation", dest="annotation", defaul=None, help="Partial annotations of source and target")
    optParser.add_argument("-asc", "--annotationSourceColumn", dest="annotationSourceColumn", defaul=0, type=int, help="Partial annotations column number of source words in file (default=0)")
    optParser.add_argument("-atc", "--annotationTargetColumn", dest="annotationTargetColumn", defaul=1, type=int, help="Partial annotations Column number of target words in file (default=1)")
    optParser.add_argument("-l1", "--ibm1Lambda", dest="ibm1Lambda", default=0.5, type=float, help="Interpolation lambda for IBM Model 1 (default=0.5)")
    optParser.add_argument("-i1", "--interIBMModel1", dest="interIBMModel1", action='store_true', default=False, help="Should interpolate IBM Model 1 or not (default=False)")
    optParser.add_argument("-l2", "--ibm2Lambda", dest="ibm2Lambda", default=0.5, type=float, help="Interpolation lambda for IBM Model 2 (default=0.5)")
    optParser.add_argument("-i2", "--interIBMModel2", dest="interIBMModel2", action='store_true', default=False, help="Should interpolate IBM Model 2 or not (default=False)")
    optParser.add_argument("-l3", "--hmmLambda", dest="hmmLambda", default=0.5, type=float, help="Interpolation lambda for HMM (default=0.5)")
    optParser.add_argument("-i3", "--interHMM", dest="interHMM", action='store_true', default=False, help="Should interpolate HMM or not (default=False)")
    optParser.add_argument("--ibm2", dest="ibm2", action='store_true', default=False, help="Turn on IBM Model 2 (default=False)")
    optParser.add_argument("--hmm", dest="hmm", action='store_true', default=False, help="Turn on HMM (default=False")
    optParser.add_argument("-o", "--output", dest="alignmentFile", defaul=None, help="Output alignments")
    optParser.add_argument("-g", "--gold", dest="goldAlignments", defaul=None, help="Gold alignments for testing")

    opts = optParser.parse_args()

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

    partialAlignments = readPartialAlignments(opts.annotation, opts.annotationSourceColumn, opts.annotationTargetColumn)

    # Run IBM Model 1
    tProb, qProb = runIBMModel1(feCount, bitextFE, partialAlignments, opts.ibm1Lambda, opts.interIBMModel1)

    # Run IBM Model 2
    if opts.ibm2:
        tProb, qProb = runIBMModel2(feCount, bitextFE, tProb, qProb, partialAlignments, opts.ibm2Lambda,
                                    opts.interIBMModel2)

    # Run HMM
    if opts.hmm:
        transitionMatrix, emissionMatrix, stateDistribution = runHMM(fCount, feCount, bitextFE, tProb,
                                                                     partialAlignments, opts.hmmLambda, opts.interHMM)
        # Find best alignments
        findBestAlignmentsForAll_AER(bitextTest, transitionMatrix, emissionMatrix, stateDistribution, 100,
                                     opts.alignmentFile)
        grade_align(fData, eData, opts.goldAlignments, opts.alignmentFile, sys.stdout)
    else:
        if opts.ibm2:
            alignmentFromIBM2(bitextFE, tProb, qProb, opts.alignmentFile)
        else:
            alignmentFromIBM1(bitextFE, tProb, opts.alignmentFile)