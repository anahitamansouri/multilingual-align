__author__ = 'Jasneet Sabharwal <jsabharw@sfu.ca>'

from collections import defaultdict


def maximizationTProb(stCoOccurrenceCount, stCounts, tCounts, tProb):
    for (sWord, tWord) in stCoOccurrenceCount:
        try:
            tProb[(sWord, tWord)] = stCounts[(sWord, tWord)] / tCounts[tWord]
        except ZeroDivisionError:
            pass
    return tProb


def maximizationInterpolatedTProb(stCoOccurrenceCount, stCounts, tCounts, tProb, supervisedTProb, lWeight):
    for sWord, tWord in stCoOccurrenceCount:
        try:
            tProbST = stCounts[(sWord, tWord)] / tCounts[tWord]
        except ZeroDivisionError:
            tProbST = 0.0
        supervisedTProbST = supervisedTProb[(sWord, tWord)]
        tProb[(sWord, tWord)] = (lWeight * supervisedTProbST) + ((1 - lWeight) * tProbST)
    return tProb


def maximizationQProb(qProb, jiCounts, iCounts, jValues, iValues, lValues, mValues):
    for j in jValues:
        for i in iValues:
            for l in lValues:
                for m in mValues:
                    try:
                        qProb[(j, i, l, m)] = jiCounts[(j, i, l, m)] / iCounts[(i, l, m)]
                    except ZeroDivisionError:
                        pass
    return qProb


def maximizationInterpolatedQProb(qProb, jiCounts, iCounts, jValues, iValues, lValues, mValues, supervisedQProb,
                                  lWeight):
    for j in jValues:
        for i in iValues:
            for l in lValues:
                for m in mValues:
                    try:
                        qProbJILM = jiCounts[(j, i, l, m)] / iCounts[(i, l, m)]
                    except ZeroDivisionError:
                        qProbJILM = 0.0
                    supervisedQProbJILM = supervisedQProb[(j, i, l, m)]
                    qProb[(j, i, l, m)] = (lWeight * supervisedQProbJILM) + ((1 - lWeight) * qProbJILM)
    return qProb


def initializeTProbUniformly(sourceCounts, stCoOccurrenceCount):
    tProb = defaultdict(float)
    for (source, target) in stCoOccurrenceCount:
        tProb[(source, target)] = 1.0/len(sourceCounts)
    return tProb


def initializeQProbUniformly(bitext):
    qProb = defaultdict(float)

    lValues = set()
    mValues = set()
    jValues = set()
    iValues = set()

    # Initialize qProb uniformly
    for source, target in bitext:
        lValues.add(len(target))
        mValues.add(len(source))
        for sIdx, sWord in enumerate(source):
            iValues.add(sIdx)
            for tIdx, tWord in enumerate(target):
                jValues.add(tIdx)

    uniformVal = max(len(iValues), len(lValues), len(mValues))
    for j in jValues:
        for i in iValues:
            for l in lValues:
                for m in mValues:
                    qProb[(j, i, l, m)] = 1.0/uniformVal

    return qProb, jValues, iValues, lValues, mValues


def initializeCounts():
    stCounts = defaultdict(float)
    tCounts = defaultdict(float)
    jiCounts = defaultdict(float)
    iCounts = defaultdict(float)
    return stCounts, tCounts, jiCounts, iCounts