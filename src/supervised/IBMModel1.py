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

