# -*- coding: utf-8 -*-
from __future__ import division

__author__ = 'Jasneet Sabharwal <jsabharw@sfu.ca>'

from collections import defaultdict


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
    tProb = defaultdict(int)

    stCounts = defaultdict(int)
    tCounts = defaultdict(int)

    # Collect counts
    for (source, target) in bitext:
        for sWord in source:
            if sWord in partialAlignments:
                for tWord in target:
                    updateValue = 0
                    if tWord in partialAlignments[sWord]:
                        updateValue = 1
                    stCounts[(sWord, tWord)] += updateValue
                    tCounts[tWord] += updateValue

    # Calculate probability
    for (sWord, tWord) in stCoOccurrenceCount:
        try:
            tProb[(sWord, tWord)] = stCounts[(sWord, tWord)] / tCounts[tWord]
        except ZeroDivisionError:
            pass

    return tProb


def interpolatedIBMModel1(sVocabCount, tVocabCount, stCoOccurrenceCount, bitext, supervisedTProb, lWeight):
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

    tProb = defaultdict(int)

    # Initialize tProb uniformly
    for (source, target) in stCoOccurrenceCount:
        tProb[(source, target)] = 1.0/len(stCoOccurrenceCount)

    for emIter in range(10):
        stCounts = defaultdict(int)
        tCounts = defaultdict(int)

        # Collect counts
        for (source, target) in bitext:
            for sWord in source:
                normalizer = 0

                for tWord in target:
                    normalizer += tProb[(sWord, tWord)]

                for tWord in target:
                    updateValue = tProb[(sWord, tWord)]/normalizer
                    stCounts[(sWord, tWord)] += updateValue
                    tCounts[tWord] += updateValue

        # Calculate and interpolate probabilities
        for (sWord, tWord) in stCoOccurrenceCount:
            tProbST = stCounts[(sWord, tWord)] / tCounts[tWord]
            supervisedTProbST = supervisedTProb[(sWord, tWord)]
            tProb[(sWord, tWord)] = (lWeight * supervisedTProbST) + ((1 - lWeight) * tProbST)

    return tProb


def unsupervisedIBMModel1(sVocabCount, tVocabCount, stCoOccurrenceCount, bitext):
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
    tProb = defaultdict(int) #translation probability

    # Uniform initialization of tProb
    # TODO: Could try stCoOccurrenceCount
    for (source, target) in stCoOccurrenceCount.iterkeys():
        tProb[(source, target)] = 1.0/len(sVocabCount)

    for emIter in range(10):
        stCounts = defaultdict(int)
        tCounts = defaultdict(int)

        # Collect counts
        for (source, target) in bitext:
            for sWord in source:
                normalizer = 0.0

                for tWord in target:
                    normalizer += tProb[(sWord, tWord)]

                for tWord in target:
                    updateValue = tProb[(sWord, tWord)]/normalizer
                    stCounts[(sWord, tWord)] += updateValue
                    tCounts[tWord] += updateValue

        # Calculate probability
        for (sWord, tWord) in stCoOccurrenceCount:
            tProb[(sWord, tWord)] = stCounts[(sWord, tWord)] / tCounts[tWord]

    return tProb
