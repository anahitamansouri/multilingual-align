# -*- coding: utf-8 -*-
from __future__ import division

__author__ = 'Jasneet Sabharwal <jsabharw@sfu.ca>'

from collections import defaultdict
from ibmModelCommons import maximizationTProb, maximizationQProb, initializeQProbUniformly, initializeTProbUniformly, \
    initializeCounts, maximizationInterpolatedTProb, maximizationInterpolatedQProb


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

    lValues = set()
    mValues = set()
    jValues = set()
    iValues = set()

    # Collect counts
    for (source, target) in bitext:
        for sIdx, sWord in enumerate(source):
            if sWord in partialAlignments:
                for tIdx, tWord in enumerate(target):
                    updateValue = 0.0
                    if tWord in partialAlignments[sWord]:
                        iValues.add(tIdx)
                        updateValue = 1.0
                    stCounts[(sWord, tWord)] += updateValue
                    tCounts[tWord] += updateValue
                    jiCounts[(sIdx, tIdx, len(target), len(source))] += updateValue
                    iCounts[(tIdx, len(target), len(source))] += updateValue,
                jValues.add(sIdx)
                lValues.add(len(target))
                mValues.add(len(source))

    # Calculate probability
    tProb = maximizationTProb(stCoOccurrenceCount, stCounts, tCounts, tProb)
    qProb = maximizationQProb(qProb, jiCounts, iCounts, jValues, iValues, lValues, mValues)

    return tProb, qProb


def interpolatedIBMModel1(stCoOccurrenceCount, bitext, supervisedTProb, lWeight):
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
    tProb = initializeTProbUniformly(stCoOccurrenceCount)
    qProb, jValues, iValues, lValues, mValues = initializeQProbUniformly(bitext)

    for emIter in range(10):
        stCounts, tCounts, jiCounts, iCounts = initializeCounts()

        # Collect counts
        for (source, target) in bitext:
            lValues.add(len(target))
            mValues.add(len(source))
            for sIdx, sWord in enumerate(source):
                normalizer = 0
                jValues.add(sIdx)

                for tWord in target:
                    normalizer += tProb[(sWord, tWord)]

                for tIdx, tWord in enumerate(target):
                    iValues.add(tIdx)
                    updateValue = tProb[(sWord, tWord)]/normalizer
                    stCounts[(sWord, tWord)] += updateValue
                    tCounts[tWord] += updateValue
                    jiCounts[(sIdx, tIdx, len(target), len(source))] += updateValue
                    iCounts[(tIdx, len(target), len(source))] += updateValue

        # Calculate and interpolate probabilities
        tProb = maximizationInterpolatedTProb(stCoOccurrenceCount, stCounts, tCounts, tProb, supervisedTProb, lWeight)
        qProb = maximizationQProb(qProb, jiCounts, iCounts, jValues, iValues, lValues, mValues)

    return tProb, qProb


def unsupervisedIBMModel1(stCoOccurrenceCount, bitext):
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
    tProb = initializeTProbUniformly(stCoOccurrenceCount)

    # Initialize alignment probability uniformly
    qProb, jValues, iValues, lValues, mValues = initializeQProbUniformly(bitext)

    for emIter in range(10):
        stCounts, tCounts, jiCounts, iCounts = initializeCounts()

        # Collect counts
        for (source, target) in bitext:
            lValues.add(len(target))
            mValues.add(len(source))
            for sIdx, sWord in enumerate(source):
                normalizer = 0.0
                jValues.add(sIdx)

                for tWord in target:
                    normalizer += tProb[(sWord, tWord)]

                for tIdx, tWord in enumerate(target):
                    iValues.add(tIdx)
                    updateValue = tProb[(sWord, tWord)]/normalizer
                    stCounts[(sWord, tWord)] += updateValue
                    tCounts[tWord] += updateValue
                    jiCounts[(sIdx, tIdx, len(target), len(source))] += updateValue
                    iCounts[(tIdx, len(target), len(source))] += updateValue

        # Calculate probability
        tProb = maximizationTProb(stCoOccurrenceCount, stCounts, tCounts, tProb)
        qProb = maximizationQProb(qProb, jiCounts, iCounts, jValues, iValues, lValues, mValues)

    return tProb, qProb
