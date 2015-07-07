# -*- coding: utf-8 -*-
from __future__ import division
__author__ = 'Jasneet Sabharwal <jsabharw@sfu.ca>'

from collections import defaultdict


def supervisedIBMModel2(stCoOccurrenceCount, bitext, partialAlignments):

    stCounts = defaultdict(float)
    tCounts = defaultdict(float)
    jiCounts = defaultdict(float)
    iCounts = defaultdict(float)

    tProb = defaultdict(float)
    qProb = defaultdict(float)

    lValues = set()
    mValues = set()
    jValues = set()
    iValues = set()

    for source, target in bitext:
        for sIdx, sWord in enumerate(source):
            if sWord in partialAlignments:
                for tIdx, tWord in enumerate(target):
                    updateValue = 0.0
                    if tWord in stCoOccurrenceCount[sWord]:
                        iValues.add(tIdx) # for using later in q(j|i,l,m) normalization
                        updateValue = 1.0
                    stCounts[(sWord, tWord)] += updateValue # for using later in t(f|e) normalization
                    tCounts[tWord] += updateValue # for using later in t(f|e) normalization
                    jiCounts[(sIdx, tIdx, len(target), len(source))] += 1 # for using later in q(j|i,l,m) normalization
                    iCounts[(tIdx, len(target), len(source))] += 1 # for using later in q(j|i,l,m) normalization
                jValues.add(sIdx) # for using later in q(j|i,l,m) normalization
                lValues.add(len(target)) # for using later in q(j|i,l,m) normalization
                mValues.add(len(source)) # for using later in q(j|i,l,m) normalization

    for sWord, tWord in stCoOccurrenceCount:
        tProb[(sWord, tWord)] = stCounts[(sWord, tWord)] / tCounts[tWord]

    for j in jValues:
        for i in iValues:
            for l in lValues:
                for m in mValues:
                    qProb[(j, i, l, m)] = jiCounts[(j, i, l, m)] / iCounts[(i, l, m)]

    return tProb, qProb


def unsupervisedIBMModel2(stCoOccurrenceCount, bitext, ibm1TProb, ibm1QProb):

    tProb = ibm1TProb
    qProb = ibm1QProb

    lValues = set()
    mValues = set()
    jValues = set()
    iValues = set()

    # # Initialize qProb uniformly
    # for source, target in bitext:
    #     lValues.add(len(target))
    #     mValues.add(len(source))
    #     for sIdx, sWord in source:
    #         iValues.add(sIdx)
    #         for tIdx, tWord in target:
    #             jValues.add(tIdx)
    #
    # uniformVal = max(len(iValues), len(lValues), len(mValues))
    # for j in jValues:
    #     for i in iValues:
    #         for l in lValues:
    #             for m in mValues:
    #                 qProb[(j, i, l, m)] = 1.0/uniformVal

    for emIter in range(10):
        stCounts = defaultdict(float)
        tCounts = defaultdict(float)
        jiCounts = defaultdict(float)
        iCounts = defaultdict(float)

        for source, target in bitext:
            lValues.add(len(target))
            mValues.add(len(source))
            for sIdx, sWord in source:
                iValues.add(sIdx)

                normalizer = 0.0
                for tIdx, tWord in target:
                    jValues.add(tIdx)
                    normalizer += qProb[(tIdx, sIdx, len(target), len(source))] * tProb[(sWord, tWord)]

                for tIdx, tWord in target:
                    updateValue = (qProb[(tIdx, sIdx, len(target), len(source))] * tProb[(sWord, tWord)]) / normalizer
                    stCounts[(sWord, tWord)] += updateValue
                    tCounts[tWord] += updateValue
                    jiCounts[(sIdx, tIdx, len(target), len(source))] += updateValue
                    iCounts[(tIdx, len(target), len(source))] += updateValue

        for sWord, tWord in stCoOccurrenceCount:
            tProb[(sWord, tWord)] = stCounts[(sWord, tWord)] / tCounts[tWord]

        for j in jValues:
            for i in iValues:
                for l in lValues:
                    for m in mValues:
                        qProb[(j, i, l, m)] = jiCounts[(j, i, l, m)] / iCounts[(i, l, m)]

    return tProb, qProb


def interpolatedIBMModel2(stCoOccurrenceCount, bitext, ibm1TProb, ibm1QProb, supervisedIBM2TProb, supervisedIBM2QProb, lWeight):
    tProb = ibm1TProb
    qProb = ibm1QProb

    lValues = set()
    mValues = set()
    jValues = set()
    iValues = set()

    for emIter in range(10):
        stCounts = defaultdict(float)
        tCounts = defaultdict(float)
        jiCounts = defaultdict(float)
        iCounts = defaultdict(float)

        for source, target in bitext:
            lValues.add(len(target))
            mValues.add(len(source))
            for sIdx, sWord in source:
                iValues.add(sIdx)

                normalizer = 0.0
                for tIdx, tWord in target:
                    jValues.add(tIdx)
                    normalizer += qProb[(tIdx, sIdx, len(target), len(source))] * tProb[(sWord, tWord)]

                for tIdx, tWord in target:
                    updateValue = (qProb[(tIdx, sIdx, len(target), len(source))] * tProb[(sWord, tWord)]) / normalizer
                    stCounts[(sWord, tWord)] += updateValue
                    tCounts[tWord] += updateValue
                    jiCounts[(sIdx, tIdx, len(target), len(source))] += updateValue
                    iCounts[(tIdx, len(target), len(source))] += updateValue

        for sWord, tWord in stCoOccurrenceCount:
            tProbST = stCounts[(sWord, tWord)] / tCounts[tWord]
            supervisedTProbST = supervisedIBM2TProb[(sWord, tWord)]
            tProb[(sWord, tWord)] = (lWeight * supervisedTProbST) + ((1 - lWeight) * tProbST)

        for j in jValues:
            for i in iValues:
                for l in lValues:
                    for m in mValues:
                        qprobJILM = jiCounts[(j, i, l, m)] / iCounts[(i, l, m)]
                        supervisedQProbJILM = supervisedIBM2QProb[(j, i, l, m)]
                        qProb[(j, i, l, m)] = (lWeight * supervisedQProbJILM) + ((1 - lWeight) * qprobJILM)

    return tProb, qProb