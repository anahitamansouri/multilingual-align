# -*- coding: utf-8 -*-
from __future__ import division

__author__ = 'Jasneet Sabharwal <jsabharw@sfu.ca>'

from collections import defaultdict
from ibmModelCommons import maximizationTProb, maximizationQProb, initializeQProbUniformly, initializeTProbUniformly, \
    initializeCounts, maximizationInterpolatedTProb, maximizationInterpolatedQProb
import multiprocessing
import math


def supervisedIBMModel1(stCoOccurrenceCount, bitext, partialAlignments):
    """
    Supervised version of IBM Model 1. In this the translation probability is only counted for
    the partial annotations of alignments that we have
    :param stCoOccurrenceCount: Dictionary containing the source and target words co-occurrence count
    :param bitext: list of tuples containing the source and target sentences
    :param partialAlignments: partial alignments from the partial annotations (source: [list of aligned target words])
    :return: translation probability, t(f|e)
    """

    # translation probability
    tProb = defaultdict(float)
    qProb = defaultdict(float)

    stCounts, tCounts, jiCounts, iCounts = initializeCounts()

    jilmCombinations = []

    # Collect counts
    for (source, target) in bitext:
        for sIdx, sWord in enumerate(source):
            if sWord in partialAlignments:
                for tIdx, tWord in enumerate(target):
                    updateValue = 0.0
                    if tWord in partialAlignments[sWord]:
                        jilm = (tIdx, sIdx, len(target), len(source))
                        ilm = (sIdx, len(target), len(source))
                        jilmCombinations.append((jilm, ilm))
                        updateValue = 1.0
                    stCounts[(sWord, tWord)] += updateValue
                    tCounts[tWord] += updateValue
                    jiCounts[(tIdx, sIdx, len(target), len(source))] += updateValue
                    iCounts[(sIdx, len(target), len(source))] += updateValue

    # Calculate probability
    tProb = maximizationTProb(stCoOccurrenceCount, stCounts, tCounts, tProb)
    qProb = maximizationQProb(qProb, jiCounts, iCounts, jilmCombinations)

    return tProb, qProb


def interpolatedIBMModel1(sourceCounts, stCoOccurrenceCount, bitext, supervisedTProb, supervisedQProb, lWeight):
    """
    Interpolated version of IBM Model1. It is the standard unsupervised IBM Model 1 but linearly interpolates
    with supervised translation probabilities during maximization step

    :param sVocabCount: source vocab occurrence count
    :param tVocabCount: target vocab occurrence count
    :param stCoOccurrenceCount: source and target word co-occurrence count
    :param bitext: parallel source and target sentences
    :param supervisedTProb: translation probability, t(f|e), from supervised IBM Model 1
    :param lWeight: lambda for linear interpolation of two models
    :return: translation probability, t(f|e)
    """

    # Initialize tProb uniformly
    tProb = initializeTProbUniformly(sourceCounts, stCoOccurrenceCount)
    qProb, jilmCombinations = initializeQProbUniformly(bitext)

    for emIter in range(10):

        # Collect counts
        stCounts, tCounts, jiCounts, iCounts = _expectation(bitext, tProb)

        # Calculate and interpolate probabilities
        tProb = maximizationInterpolatedTProb(stCoOccurrenceCount, stCounts, tCounts, tProb, supervisedTProb, lWeight)
        qProb = maximizationInterpolatedQProb(qProb, jiCounts, iCounts, jilmCombinations, supervisedQProb, lWeight)

    return tProb, qProb


def unsupervisedIBMModel1(sourceCounts, stCoOccurrenceCount, bitext):
    """
    Unsupervised version of IBM Model 1. In this the translation probability is only counted for
    the partial annotations of alignments that we have
    :param sourceVocabCount: Dictionary containing the source vocab and their occurrence count
    :param targetVocabCount: Dictionary containing the target vocab and their occurrence count
    :param stCoOccurrenceCount: Dictionary containing the source and target words co-occurrence count
    :param bitext: list of tuples containing the source and target sentences
    :param partialAlignments: partial alignments from the partial annotations (source: [list of target words])
    :return: translation probability, t(f|e)
    """
    # Initialize translation probability uniformly
    tProb = initializeTProbUniformly(sourceCounts, stCoOccurrenceCount)

    # Initialize alignment probability uniformly
    qProb, jilmCombinations = initializeQProbUniformly(bitext)

    for emIter in range(10):

        # Calculate Counts
        stCounts, tCounts, jiCounts, iCounts = _expectation(bitext, tProb)

        # Calculate probability
        tProb = maximizationTProb(stCoOccurrenceCount, stCounts, tCounts, tProb)
        qProb = maximizationQProb(qProb, jiCounts, iCounts, jilmCombinations)

    return tProb, qProb


def parallelInterpolatedIBMModel1(sourceCounts, stCoOccurrenceCount, bitext, supervisedTProb, supervisedQProb, lWeight):
    """
    Interpolated version of IBM Model1. It is the standard unsupervised IBM Model 1 but linearly interpolates
    with supervised translation probabilities during maximization step

    :param sVocabCount: source vocab occurrence count
    :param tVocabCount: target vocab occurrence count
    :param stCoOccurrenceCount: source and target word co-occurrence count
    :param bitext: parallel source and target sentences
    :param supervisedTProb: translation probability, t(f|e), from supervised IBM Model 1
    :param lWeight: lambda for linear interpolation of two models
    :return: translation probability, t(f|e)
    """

    # Initialize tProb uniformly
    tProb = initializeTProbUniformly(sourceCounts, stCoOccurrenceCount)
    qProb, jilmCombinations = initializeQProbUniformly(bitext)

    for emIter in range(10):
        stCounts, tCounts, jiCounts, iCounts = initializeCounts()

        # Collect counts
        outputQueue = multiprocessing.Queue()
        numberOfProcessAllowed = multiprocessing.cpu_count()
        chunkSize = int(math.ceil(len(bitext) / float(numberOfProcessAllowed)))
        procs = []

        for procNum in range(numberOfProcessAllowed):
            bitextChunk = bitext[chunkSize*procNum:chunkSize*(procNum+1)]
            proc = multiprocessing.Process(target=_parallelExpectation, args=(bitextChunk, tProb,
                                                                              outputQueue))
            procs.append(proc)
            proc.start()

        for procNum in range(numberOfProcessAllowed):
            partialResult = outputQueue.get()
            stCounts = _mergeCounts(partialResult['stCounts'], stCounts)
            tCounts = _mergeCounts(partialResult['tCounts'], tCounts)
            jiCounts = _mergeCounts(partialResult['jiCounts'], jiCounts)
            iCounts = _mergeCounts(partialResult['iCounts'], iCounts)

        for proc in procs:
            proc.join()

        # Calculate and interpolate probabilities
        tProb = maximizationInterpolatedTProb(stCoOccurrenceCount, stCounts, tCounts, tProb, supervisedTProb, lWeight)
        qProb = maximizationInterpolatedQProb(qProb, jiCounts, iCounts, jilmCombinations, supervisedQProb, lWeight)

    return tProb, qProb


def parallelUnsupervisedIBMModel1(sourceCounts, stCoOccurrenceCount, bitext):
    # Initialize translation probability uniformly
    tProb = initializeTProbUniformly(sourceCounts, stCoOccurrenceCount)

    # Initialize alignment probability uniformly
    qProb, jilmCombinations = initializeQProbUniformly(bitext)

    for emIter in range(10):
        print ">>>> IBM Model 1 - EM Iteration " + str(emIter)
        stCounts, tCounts, jiCounts, iCounts = initializeCounts()

        # Collect counts
        print '>>>> >>>> Expectation Step'
        outputQueue = multiprocessing.Queue()
        numberOfProcessAllowed = multiprocessing.cpu_count()
        chunkSize = int(math.ceil(len(bitext) / float(numberOfProcessAllowed)))
        procs = []

        for procNum in range(numberOfProcessAllowed):
            bitextChunk = bitext[chunkSize*procNum:chunkSize*(procNum+1)]
            proc = multiprocessing.Process(target=_parallelExpectation, args=(bitextChunk, tProb, outputQueue))
            procs.append(proc)
            proc.start()

        for procNum in range(numberOfProcessAllowed):
            partialResult = outputQueue.get()
            stCounts = _mergeCounts(partialResult['stCounts'], stCounts)
            tCounts = _mergeCounts(partialResult['tCounts'], tCounts)
            jiCounts = _mergeCounts(partialResult['jiCounts'], jiCounts)
            iCounts = _mergeCounts(partialResult['iCounts'], iCounts)

        for proc in procs:
            proc.join()

        # Calculate probability
        print '>>>> >>>> Maximizing tProb'
        tProb = maximizationTProb(stCoOccurrenceCount, stCounts, tCounts, tProb)
        print '>>>> >>>> Maximizing qProb'
        qProb = maximizationQProb(qProb, jiCounts, iCounts, jilmCombinations)

    return tProb, qProb


def _expectation(bitext, tProb):
    stCounts, tCounts, jiCounts, iCounts = initializeCounts()

    # Collect counts
    for (source, target) in bitext:
        for sIdx, sWord in enumerate(source):
            normalizer = 0.0

            for tWord in target:
                normalizer += tProb[(sWord, tWord)]

            for tIdx, tWord in enumerate(target):
                updateValue = tProb[(sWord, tWord)]/normalizer
                stCounts[(sWord, tWord)] += updateValue
                tCounts[tWord] += updateValue
                jiCounts[(tIdx, sIdx, len(target), len(source))] += updateValue
                iCounts[(sIdx, len(target), len(source))] += updateValue

    return stCounts, tCounts, jiCounts, iCounts


def _parallelExpectation(bitext, tProb, outputQueue):
    stCounts, tCounts, jiCounts, iCounts = _expectation(bitext, tProb)
    outputQueue.put({'stCounts': stCounts, 'tCounts': tCounts, 'jiCounts': jiCounts, 'iCounts': iCounts})


def _mergeCounts(partialCountDict, countDict):
    for key, value in partialCountDict.iteritems():
        countDict[key] += value
    return countDict
