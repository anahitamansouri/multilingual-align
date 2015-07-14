__author__ = 'Jasneet Sabharwal <jsabharw@sfu.ca>'

from collections import defaultdict


def maximizationTProb(stCoOccurrenceCount, stCounts, tCounts, tProb):
    print '>>>> Maximizing TProb'
    for (sWord, tWord) in stCoOccurrenceCount:
        try:
            tProb[(sWord, tWord)] = stCounts[(sWord, tWord)] / tCounts[tWord]
        except ZeroDivisionError:
            pass
    return tProb


def maximizationInterpolatedTProb(stCoOccurrenceCount, stCounts, tCounts, tProb, supervisedTProb, lWeight):
    print '>>>> Maximizing Interpolated TProb'
    for sWord, tWord in stCoOccurrenceCount:
        try:
            tProbST = stCounts[(sWord, tWord)] / tCounts[tWord]
        except ZeroDivisionError:
            tProbST = 0.0
        supervisedTProbST = supervisedTProb[(sWord, tWord)]
        tProb[(sWord, tWord)] = (lWeight * supervisedTProbST) + ((1 - lWeight) * tProbST)
    return tProb


#def maximizationQProb(qProb, jiCounts, iCounts, jValues, iValues, lValues, mValues):
#    print '>>>> Maximizing QProb'
#    for j in jValues:
#        for i in iValues:
#            for l in lValues:
#                for m in mValues:
#                    try:
#                        qProb[(j, i, l, m)] = jiCounts[(j, i, l, m)] / iCounts[(i, l, m)]
#                    except ZeroDivisionError:
#                        qProb[(j, i, l, m)] = 0.0
#    return qProb

def maximizationQProb(qProb, jiCounts, iCounts, jilmCombinations):
    for jilm, ilm in jilmCombinations:
        try:
            qProb[jilm] = jiCounts[jilm] / iCounts[ilm]
        except ZeroDivisionError:
            qProb[jilm] = 0.0
    return qProb


#def maximizationInterpolatedQProb(qProb, jiCounts, iCounts, jValues, iValues, lValues, mValues, supervisedQProb,
#                                  lWeight):
#    print '>>>> Maximizing Interpolated QProb'
#    for j in jValues:
#        for i in iValues:
#            for l in lValues:
#                for m in mValues:
#                    try:
#                        qProbJILM = jiCounts[(j, i, l, m)] / iCounts[(i, l, m)]
#                    except ZeroDivisionError:
#                        qProbJILM = 0.0
#                    supervisedQProbJILM = supervisedQProb[(j, i, l, m)]
#                    qProb[(j, i, l, m)] = (lWeight * supervisedQProbJILM) + ((1 - lWeight) * qProbJILM)
#    return qProb


def maximizationInterpolatedQProb(qProb, jiCounts, iCounts, jilmCombinations, supervisedQProb, lWeight):
    for jilm, ilm in jilmCombinations:
        try:
            qProbJILM = jiCounts[jilm] / iCounts[ilm]
        except ZeroDivisionError:
            qProb[jilm] = 0.0
        supervisedQProbJILM = supervisedQProb[jilm]
        qProb[jilm] = (lWeight * supervisedQProbJILM) + ((1 - lWeight) * qProbJILM)
    return qProb


def initializeTProbUniformly(sourceCounts, stCoOccurrenceCount):
    print '>>>> Initializing TProb Uniformly'
    tProb = defaultdict(float)
    for (source, target) in stCoOccurrenceCount:
        tProb[(source, target)] = 1.0/len(sourceCounts)
    return tProb


def initializeQProbUniformly(bitext):
    print '>>>> Initializing QProb Uniformly'
    qProb = defaultdict(float)

    jValues = set()

    jilmCombinations = []

    # Initialize qProb uniformly
    for n, (source, target) in enumerate(bitext):
        if n % 1000 == 0:
            print '>>>> >>>> {}'.format(n)
        for sIdx, sWord in enumerate(source):
            for tIdx, tWord in enumerate(target):
                jValues.add(tIdx)
                jilm = (tIdx, sIdx, len(target), len(source))
                ilm = (sIdx, len(target), len(source))
                jilmCombinations.append((jilm, ilm))

    #uniformVal = max(len(iValues), len(lValues), len(mValues))
    uniformVal = len(jValues)
    for jilm, ilm in jilmCombinations:
        qProb[jilm] = 1.0/uniformVal

    return qProb, jilmCombinations


def initializeCounts():
    stCounts = defaultdict(float)
    tCounts = defaultdict(float)
    jiCounts = defaultdict(float)
    iCounts = defaultdict(float)
    return stCounts, tCounts, jiCounts, iCounts