# -*- coding: utf-8 -*-
from __future__ import division
__author__ = 'Jasneet Sabharwal <jsabharw@sfu.ca>'

from collections import defaultdict
from ibmModelCommons import initializeCounts, maximizationQProb, maximizationTProb, maximizationInterpolatedTProb, \
    maximizationInterpolatedQProb, initializeQProbUniformly
import multiprocessing
import math


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
                        jValues.add(tIdx) # for using later in q(j|i,l,m) normalization
                        updateValue = 1.0
                    stCounts[(sWord, tWord)] += updateValue # for using later in t(f|e) normalization
                    tCounts[tWord] += updateValue # for using later in t(f|e) normalization
                    jiCounts[(tIdx, sIdx, len(target), len(source))] += 1 # for using later in q(j|i,l,m) normalization
                    iCounts[(sIdx, len(target), len(source))] += 1 # for using later in q(j|i,l,m) normalization
                iValues.add(sIdx) # for using later in q(j|i,l,m) normalization
                lValues.add(len(target)) # for using later in q(j|i,l,m) normalization
                mValues.add(len(source)) # for using later in q(j|i,l,m) normalization

    tProb = maximizationTProb(stCoOccurrenceCount, stCounts, tCounts, tProb)
    qProb = maximizationQProb(qProb, jiCounts, iCounts, jValues, iValues, lValues, mValues)

    return tProb, qProb


def unsupervisedIBMModel2(stCoOccurrenceCount, bitext, ibm1TProb, ibm1QProb):

    tProb = ibm1TProb
    qProb = ibm1QProb

    _, jValues, iValues, lValues, mValues = initializeQProbUniformly(bitext)

    for emIter in range(10):
        # Calculate counts
        stCounts, tCounts, jiCounts, iCounts = _expectation(bitext, tProb, qProb)

        # Calculate probabilities
        tProb = maximizationTProb(stCoOccurrenceCount, stCounts, tCounts, tProb)
        qProb = maximizationQProb(qProb, jiCounts, iCounts, jValues, iValues, lValues, mValues)

    return tProb, qProb


def interpolatedIBMModel2(stCoOccurrenceCount, bitext, ibm1TProb, ibm1QProb, supervisedIBM2TProb, supervisedIBM2QProb,
                          lWeight):
    tProb = ibm1TProb
    qProb = ibm1QProb

    _, jValues, iValues, lValues, mValues = initializeQProbUniformly(bitext)

    for emIter in range(10):

        # Calculate counts
        stCounts, tCounts, jiCounts, iCounts = _expectation(bitext, tProb, qProb)

        # Calculate probabilities
        tProb = maximizationInterpolatedTProb(stCoOccurrenceCount, stCounts, tCounts, tProb, supervisedIBM2TProb,
                                              lWeight)
        qProb = maximizationInterpolatedQProb(qProb, jiCounts, iCounts, jValues, iValues, lValues, mValues,
                                              supervisedIBM2QProb, lWeight)

    return tProb, qProb


def parallelUnsupervisedIBMModel2(stCoOccurrenceCount, bitext, ibm1TProb, ibm1QProb):

    tProb = ibm1TProb
    qProb = ibm1QProb

    _, jValues, iValues, lValues, mValues = initializeQProbUniformly(bitext)

    for emIter in range(10):
        stCounts, tCounts, jiCounts, iCounts = initializeCounts()

        # Calculate counts
        outputQueue = multiprocessing.Queue()
        numberOfProcessAllowed = multiprocessing.cpu_count()
        chunkSize = int(math.ceil(len(bitext) / float(numberOfProcessAllowed)))
        procs = []

        for procNum in range(numberOfProcessAllowed):
            bitextChunk = bitext[chunkSize*procNum:chunkSize*(procNum+1)]
            proc = multiprocessing.Process(target=_parallelExpectation, args=(bitextChunk, tProb, qProb, outputQueue))
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

        # Calculate probabilities
        tProb = maximizationTProb(stCoOccurrenceCount, stCounts, tCounts, tProb)
        qProb = maximizationQProb(qProb, jiCounts, iCounts, jValues, iValues, lValues, mValues)

    return tProb, qProb


def parallelInterpolatedIBMModel2(stCoOccurrenceCount, bitext, ibm1TProb, ibm1QProb, supervisedIBM2TProb,
                                  supervisedIBM2QProb, lWeight):
    tProb = ibm1TProb
    qProb = ibm1QProb

    _, jValues, iValues, lValues, mValues = initializeQProbUniformly(bitext)

    for emIter in range(10):
        print ">>>> IBM Model 1 - EM Iteration " + str(emIter)

        stCounts, tCounts, jiCounts, iCounts = initializeCounts()

        # Calculate counts
        outputQueue = multiprocessing.Queue()
        numberOfProcessAllowed = multiprocessing.cpu_count()
        chunkSize = int(math.ceil(len(bitext) / float(numberOfProcessAllowed)))
        procs = []

        for procNum in range(numberOfProcessAllowed):
            bitextChunk = bitext[chunkSize*procNum:chunkSize*(procNum+1)]
            proc = multiprocessing.Process(target=_parallelExpectation, args=(bitextChunk, tProb, qProb, outputQueue))
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

        # Calculate probabilities
        tProb = maximizationInterpolatedTProb(stCoOccurrenceCount, stCounts, tCounts, tProb, supervisedIBM2TProb,
                                              lWeight)
        qProb = maximizationInterpolatedQProb(qProb, jiCounts, iCounts, jValues, iValues, lValues, mValues,
                                              supervisedIBM2QProb, lWeight)

    return tProb, qProb


def _expectation(bitext, tProb, qProb):
    stCounts, tCounts, jiCounts, iCounts = initializeCounts()

    for source, target in bitext:
        for sIdx, sWord in enumerate(source):
            normalizer = 0.0

            for tIdx, tWord in enumerate(target):
                normalizer += qProb[(tIdx, sIdx, len(target), len(source))] * tProb[(sWord, tWord)]

            for tIdx, tWord in enumerate(target):
                updateValue = (qProb[(tIdx, sIdx, len(target), len(source))] * tProb[(sWord, tWord)]) / normalizer
                stCounts[(sWord, tWord)] += updateValue
                tCounts[tWord] += updateValue
                jiCounts[(tIdx, sIdx, len(target), len(source))] += updateValue
                iCounts[(sIdx, len(target), len(source))] += updateValue

    return stCounts, tCounts, jiCounts, iCounts


def _parallelExpectation(bitext, tProb, qProb, outputQueue):
    stCounts, tCounts, jiCounts, iCounts = _expectation(bitext, tProb, qProb)
    outputQueue.put({'stCounts': stCounts, 'tCounts': tCounts, 'jiCounts': jiCounts, 'iCounts': iCounts})


def _mergeCounts(partialCountDict, countDict):
    for key, value in partialCountDict.iteritems():
        countDict[key] += value
    return countDict