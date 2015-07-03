# -*- coding: utf-8 -*-
__author__ = 'Jasneet Sabharwal <jsabharw@sfu.ca>'

from collections import defaultdict
from numpy import sum, zeros


def supervisedHMM(stCoOccurrenceCount, bitext, partialAlignments):

    tProb = defaultdict(int)
    stCounts = defaultdict(int)
    tCounts = defaultdict(int)

    N = _getMaxTargetSentenceLength(bitext)
    A = zeros((N+1, N+1, N+1))

    for (source, target) in bitext:
        for sWordIdx, sWord in enumerate(source):
            if sWord in partialAlignments:
                prevAlignment = -1
                for tWordIdx, tWord in enumerate(target):
                    if tWord in partialAlignments[sWord]:
                        stCounts[(sWord, tWord)] += 1.0
                        tCounts[tWord] += 1.0
                        if prevAlignment == -1:
                            prevAlignment = tWordIdx+1
                        else:
                            A[prevAlignment, tWordIdx+1, len(target)] += 1.0
                            prevAlignment = tWordIdx+1

    # Calculate tProb
    for (sWord, tWord) in stCoOccurrenceCount:
        try:
            tProb[(sWord, tWord)] = stCounts[(sWord, tWord)] / tCounts[tWord]
        except ZeroDivisionError:
            pass

    # Normalize A
    A = A/sum(A, axis=0, keepdims=True)

    return A, tProb


def _getMaxTargetSentenceLength(bitext):
    """
    Get max length of target sentences
    :param bitext:
    :return: max target sentence length, target sentence lengths present in bitext
    """
    maxTLength = 0
    for source, target in bitext:
        targetLength = len(target)
        if targetLength > maxTLength:
            maxTLength = len(target)
    return maxTLength