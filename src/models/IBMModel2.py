# -*- coding: utf-8 -*-
from __future__ import division
__author__ = 'Jasneet Sabharwal <jsabharw@sfu.ca>'

from collections import defaultdict
from ibmModelCommons import initializeCounts, maximizationQProb, maximizationTProb, maximizationInterpolatedTProb, \
    maximizationInterpolatedQProb


def supervisedIBMModel2(stCoOccurrenceCount, bitext, partialAlignments):

    tProb = defaultdict(float)
    qProb = defaultdict(float)

    stCounts, tCounts, jiCounts, iCounts = initializeCounts()

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

    tProb = maximizationTProb(stCoOccurrenceCount, stCounts, tCounts, tProb)
    qProb = maximizationQProb(qProb, jiCounts, iCounts, jValues, iValues, lValues, mValues)

    return tProb, qProb


def unsupervisedIBMModel2(stCoOccurrenceCount, bitext, ibm1TProb, ibm1QProb):

    tProb = ibm1TProb
    qProb = ibm1QProb

    lValues = set()
    mValues = set()
    jValues = set()
    iValues = set()

    for emIter in range(10):
        stCounts, tCounts, jiCounts, iCounts = initializeCounts()

        for source, target in bitext:
            lValues.add(len(target))
            mValues.add(len(source))
            for sIdx, sWord in enumerate(source):
                jValues.add(sIdx)

                normalizer = 0.0
                for tIdx, tWord in enumerate(target):
                    iValues.add(tIdx)
                    normalizer += qProb[(tIdx, sIdx, len(target), len(source))] * tProb[(sWord, tWord)]

                for tIdx, tWord in enumerate(target):
                    updateValue = (qProb[(tIdx, sIdx, len(target), len(source))] * tProb[(sWord, tWord)]) / normalizer
                    stCounts[(sWord, tWord)] += updateValue
                    tCounts[tWord] += updateValue
                    jiCounts[(sIdx, tIdx, len(target), len(source))] += updateValue
                    iCounts[(tIdx, len(target), len(source))] += updateValue

        tProb = maximizationTProb(stCoOccurrenceCount, stCounts, tCounts, tProb)
        qProb = maximizationQProb(qProb, jiCounts, iCounts, jValues, iValues, lValues, mValues)

    return tProb, qProb


def interpolatedIBMModel2(stCoOccurrenceCount, bitext, ibm1TProb, ibm1QProb, supervisedIBM2TProb, supervisedIBM2QProb,
                          lWeight):
    tProb = ibm1TProb
    qProb = ibm1QProb

    lValues = set()
    mValues = set()
    jValues = set()
    iValues = set()

    for emIter in range(10):
        stCounts, tCounts, jiCounts, iCounts = initializeCounts()

        for source, target in bitext:
            lValues.add(len(target))
            mValues.add(len(source))
            for sIdx, sWord in enumerate(source):
                jValues.add(sIdx)

                normalizer = 0.0
                for tIdx, tWord in enumerate(target):
                    iValues.add(tIdx)
                    normalizer += qProb[(tIdx, sIdx, len(target), len(source))] * tProb[(sWord, tWord)]

                for tIdx, tWord in enumerate(target):
                    updateValue = (qProb[(tIdx, sIdx, len(target), len(source))] * tProb[(sWord, tWord)]) / normalizer
                    stCounts[(sWord, tWord)] += updateValue
                    tCounts[tWord] += updateValue
                    jiCounts[(sIdx, tIdx, len(target), len(source))] += updateValue
                    iCounts[(tIdx, len(target), len(source))] += updateValue

        tProb = maximizationInterpolatedTProb(stCoOccurrenceCount, stCounts, tCounts, tProb, supervisedIBM2TProb,
                                              lWeight)
        qProb = maximizationInterpolatedQProb(qProb, jiCounts, iCounts, jValues, iValues, lValues, mValues,
                                              supervisedIBM2QProb, lWeight)

    return tProb, qProb