# -*- coding: utf-8 -*-

__author__ = 'Jasneet Sabharwal <jsabharw@sfu.ca>'

import argparse
import sys
import os
import time
from collections import defaultdict
import cPickle as pickle

from tools.fileReaders import readPartialAlignments
from models.IBMModel1 import supervisedIBMModel1, interpolatedIBMModel1, unsupervisedIBMModel1, parallelUnsupervisedIBMModel1
from models.IBMModel2 import supervisedIBMModel2, interpolatedIBMModel2, unsupervisedIBMModel2, parallelUnsupervisedIBMModel2
from models.HMM import supervisedHMM
from models.HMM_with_length_with_array_null import interpolatedBaumWelchP, findBestAlignmentsForAll_AER
from unsupervised.HMM_with_length_with_array_null import baumWelchP


def runIBMModel1(sourceCounts, stCoOccurrenceCount, bitext, partialAlignments, interpolationWeight,
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
    print "\nEXECUTING IBM MODEL 1"
    startTime = time.time()

    if interpolate:
        tProb, qProb = supervisedIBMModel1(stCoOccurrenceCount, bitext, partialAlignments)
        tProb, qProb = interpolatedIBMModel1(stCoOccurrenceCount, bitext, tProb, interpolationWeight)
    else:
        tProb, qProb = parallelUnsupervisedIBMModel1(sourceCounts, stCoOccurrenceCount, bitext)

    endTime = time.time()
    print "RUN TIME FOR IBM MODEL 1! %.2gs" % (endTime - startTime)

    return tProb, qProb


def runIBMModel2(sourceCounts, stCoOccurrenceCount, bitext, ibm1TProb, ibm1QProb, partialAlignments, interpolationWeight,
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
    print "\nEXECUTING IBM MODEL 2"
    if interpolate:
        tProb, qProb = supervisedIBMModel2(stCoOccurrenceCount, bitext, partialAlignments)
        tProb, qProb = interpolatedIBMModel2(stCoOccurrenceCount, bitext, ibm1TProb, ibm1QProb, tProb, qProb,
                                             interpolationWeight)
    else:
        tProb, qProb = parallelUnsupervisedIBMModel2(stCoOccurrenceCount, bitext, ibm1TProb, ibm1QProb)

    endTime = time.time()
    print "RUN TIME FOR IBM MODEL 2! %.2gs" % (endTime - startTime)

    return tProb, qProb


def runHMM(sourceVocabCount, stCoOccurrenceCount, bitext, ibmTProb, partialAlignments, interpolationWeight,
           interpolate=True):
    startTime = time.time()
    print "\nEXECUTING HMM"

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


def printArgs(opts):
    print '#'*10 + ' ARGUMENTS ' + '#'*10
    print 'Data: {}'.format(opts.train)
    print 'Target Suffix: {}'.format(opts.english)
    print 'Source Suffix: {}'.format(opts.french)
    print 'Number of Training Sentences: {}'.format(opts.num_sents)
    print 'Partial Annotation File: {}'.format(opts.annotation)
    print 'Partial Annotation Source Column: {}'.format(opts.annotationSourceColumn)
    print 'Partial Annotation Target Column: {}'.format(opts.annotationTargetColumn)
    print 'Interpolate IBM Model 1: {}'.format(opts.interIBMModel1)
    print 'IBM Model 1 Lambda: {}'.format(opts.ibm1Lambda)
    print 'Run IBM Model 2: {}'.format(opts.ibm2)
    print 'Interpolate IBM Model 2: {}'.format(opts.interIBMModel2)
    print 'IBM Model 2 Lambda: {}'.format(opts.ibm2Lambda)
    print 'Run HMM: {}'.format(opts.hmm)
    print 'Interpolate HMM: {}'.format(opts.interHMM)
    print 'HMM Lambda: {}'.format(opts.hmmLambda)


if __name__ == '__main__':
    optParser = argparse.ArgumentParser()
    optParser.add_argument("-p", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
    optParser.add_argument("-e", "--english", dest="english", default="e", help="Suffix of English (Target) filename (default=e)")
    optParser.add_argument("-f", "--french", dest="french", default="f", help="Suffix of French (Source) filename (default=f)")
    optParser.add_argument("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type=int, help="Number of sentences to use for training and alignment")
    optParser.add_argument("-a", "--annotation", dest="annotation", default=None, help="Partial annotations of source and target")
    optParser.add_argument("-asc", "--annotationSourceColumn", dest="annotationSourceColumn", default=0, type=int, help="Partial annotations column number of source words in file (default=0)")
    optParser.add_argument("-atc", "--annotationTargetColumn", dest="annotationTargetColumn", default=1, type=int, help="Partial annotations Column number of target words in file (default=1)")
    optParser.add_argument("-l1", "--ibm1Lambda", dest="ibm1Lambda", default=0.5, type=float, help="Interpolation lambda for IBM Model 1 (default=0.5)")
    optParser.add_argument("-i1", "--interIBMModel1", dest="interIBMModel1", action='store_true', default=False, help="Should interpolate IBM Model 1 or not (default=False)")
    optParser.add_argument("-l2", "--ibm2Lambda", dest="ibm2Lambda", default=0.5, type=float, help="Interpolation lambda for IBM Model 2 (default=0.5)")
    optParser.add_argument("-i2", "--interIBMModel2", dest="interIBMModel2", action='store_true', default=False, help="Should interpolate IBM Model 2 or not (default=False)")
    optParser.add_argument("-l3", "--hmmLambda", dest="hmmLambda", default=0.5, type=float, help="Interpolation lambda for HMM (default=0.5)")
    optParser.add_argument("-i3", "--interHMM", dest="interHMM", action='store_true', default=False, help="Should interpolate HMM or not (default=False)")
    optParser.add_argument("--ibm2", dest="ibm2", action='store_true', default=False, help="Turn on IBM Model 2 (default=False)")
    optParser.add_argument("--hmm", dest="hmm", action='store_true', default=False, help="Turn on HMM (default=False")
    optParser.add_argument("-md", "--modelDir", dest="modelDir", default=None, help="Directory for storing models")
    optParser.add_argument("--ibm1-model", dest="ibm1ModelDir", default=None, help="Load pre-calculated IBM Model 1 models")
    optParser.add_argument("--ibm2-model", dest="ibm2ModelDir", default=None, help="Load pre-calculated IBM Model 2 models")

    opts = optParser.parse_args()

    printArgs(opts)

    fData = "%s.%s" % (opts.train, opts.french)
    eData = "%s.%s" % (opts.train, opts.english)

    # Check if fData and eData files exist
    if not (os.path.isfile(fData) and os.path.isfile(eData)):
        print >>sys.stderr, __doc__.strip('\n\r')
        sys.exit(1)

    sys.stderr.write('#'*10 + ' BEGINNING TRAINING ' + '#'*10 + '\n')

    # Read data from fData and eData and store them as bitextFE and bitextEF
    with open(fData) as fFile, open(eData) as eFile:
        bitextFE = []
        for pair in zip(fFile, eFile)[:opts.num_sents]:
            pairs = []
            skipSentence = False
            for sentence in pair:
                sentenceSplit = sentence.strip().split()
                pairs.append(sentenceSplit)
                if len(sentenceSplit) > 150:
                    skipSentence = True
            if not skipSentence:
                bitextFE.append(pairs)


    # Initialize variables (count variables for gathering vocab counts, co-occurrence counts)
    fCount = defaultdict(int)
    feCount = defaultdict(int)

    for (n, (f, e)) in enumerate(bitextFE):
        for f_i in set(f):
            fCount[f_i] += 1
            for e_j in e:
                feCount[(f_i, e_j)] += 1

    partialAlignments = None
    if opts.annotation:
        partialAlignments = readPartialAlignments(opts.annotation, opts.annotationSourceColumn, opts.annotationTargetColumn)

    # Run IBM Model 1
    if not opts.ibm2ModelDir:
        if opts.ibm1ModelDir:
            print "LOADING IBM MODEL 1 MODELS FROM FILE"
            with open(os.path.join(opts.ibm1ModelDir, 'ibm1.tprob')) as tProbFile, \
                    open(os.path.join(opts.ibm1ModelDir, 'ibm1.qprob')) as qProbFile:
                tProb = pickle.load(tProbFile)
                qProb = pickle.load(qProbFile)
        else:
            tProb, qProb = runIBMModel1(fCount, feCount, bitextFE, partialAlignments, opts.ibm1Lambda, opts.interIBMModel1)
            if opts.modelDir:
                with open(os.path.join(opts.modelDir, 'ibm1.tprob'), 'w') as tProbFile, \
                        open(os.path.join(opts.modelDir, 'ibm1.qprob'), 'w') as qProbFile:
                    pickle.dump(tProb, tProbFile)
                    pickle.dump(qProb, qProbFile)

    # Run IBM Model 2
    if opts.ibm2 and not opts.ibm2ModelDir:
        tProb, qProb = runIBMModel2(fCount, feCount, bitextFE, tProb, qProb, partialAlignments, opts.ibm2Lambda,
                                    opts.interIBMModel2)
        if opts.modelDir:
            with open(os.path.join(opts.modelDir, 'ibm2.tprob'), 'w') as tProbFile, \
                    open(os.path.join(opts.modelDir, 'ibm2.qprob'), 'w') as qProbFile:
                pickle.dump(tProb, tProbFile)
                pickle.dump(qProb, qProbFile)
    elif opts.ibm2 and opts.ibm2ModelDir:
        print "LOADING IBM MODEL 2 MODELS FROM FILE"
        with open(os.path.join(opts.ibm2ModelDir, 'ibm2.tprob')) as tProbFile, \
                open(os.path.join(opts.ibm2ModelDir, 'ibm2.qprob')) as qProbFile:
            tProb = pickle.load(tProbFile)
            qProb = pickle.load(qProbFile)

    # Run HMM
    if opts.hmm:
        transitionMatrix, emissionMatrix, stateDistribution = runHMM(fCount, feCount, bitextFE, tProb,
                                                                     partialAlignments, opts.hmmLambda, opts.interHMM)
        if opts.modelDir:
            with open(os.path.join(opts.modelDir, 'hmm.transition'), 'w') as transFile, \
                    open(os.path.join(opts.modelDir, 'hmm.emission'), 'w') as emisFile, \
                    open(os.path.join(opts.modelDir, 'hmm.state'), 'w') as stateFile:
                pickle.dump(transitionMatrix, transFile)
                pickle.dump(emissionMatrix, emisFile)
                pickle.dump(stateDistribution, stateFile)
