# -*- coding: utf-8 -*-
__author__ = 'Jasneet Sabharwal <jsabharw@sfu.ca>'

import argparse
import numpy as np
from scipy.spatial.distance import cosine
import multiprocessing
from os import getpid
import time

def readVectors(fileName):
    vocab = []
    vectors = []
    with open(fileName) as inFile:
        for idx, line in enumerate(inFile):
            if idx % 1000 == 0:
                print idx
            line = line.strip('\r\n')
            if line:
                lineSplit = line.split()
                word = lineSplit[0]
                vector = np.array([float(i) for i in lineSplit[1:]])
                vectors.append(vector)
                vocab.append(word)
    return np.array(vectors), vocab


def writeAlignment(fileName, sVocab, tVocab, nearestVectorsIdx):
    with open(fileName, 'w') as outFile:
        for idx, sWord in enumerate(sVocab):
            tWord = tVocab[nearestVectorsIdx[idx]]
            outFile.write('{}\t{}\n'.format(sWord, tWord))


def findClosestVector(sIdx, sVector, tVectors, returnDict):
    print 'FIND CLOSEST VECTOR - {} - pid: {}'.format(sIdx, getpid())
    smallestCosine = float("inf")
    smallestCosineIdx = -1
    for tIdx, tVector in enumerate(tVectors):
        cosineDistance = cosine(sVector, tVector)
        if cosineDistance < smallestCosine:
            smallestCosine = cosineDistance
            smallestCosineIdx = tIdx
    returnDict[sIdx] = smallestCosineIdx


def canStartMoreJobs(jobs):
    cpuCounts = multiprocessing.cpu_count()
    currentProcesses = 0
    for proc in jobs:
        if proc.is_alive():
            currentProcesses += 1

    print "---- CURRENT RUNNING PROCESS: {}".format(currentProcesses)
    print "---- CPU COUNTS: {}".format(cpuCounts)

    if currentProcesses <= cpuCounts:
        return True
    else:
        return False


if __name__ == '__main__':
    optParser = argparse.ArgumentParser()
    optParser.add_argument("-sv", "--sourceVectors", dest="sourceVectors", help="Source vectors file (example: english)")
    optParser.add_argument("-tv", "--targetVectors", dest="targetVectors", help="Target vectors file (example: chinese)")
    optParser.add_argument("-o", "--output", dest="output", help="Output alignment file")
    args = optParser.parse_args()

    print "#"*10+"Reading Vectors"+"#"*10
    sVectors, sVocab = readVectors(args.sourceVectors)
    tVectors, tVocab = readVectors(args.targetVectors)


    # test
    #sVectors = sVectors[:100]
    #sVocab = sVocab[:100]

    print "#"*10+"Calculating Cosines"+"#"*10
    manager = multiprocessing.Manager()
    nearestVectorsIdx = manager.dict()
    jobs = []

    for sIdx, sVector in enumerate(sVectors):
        print '\tCalculating Cosine for: {} - {}'.format(sIdx, sVocab[sIdx])

        proc = multiprocessing.Process(target=findClosestVector, args=(sIdx, sVector, tVectors, nearestVectorsIdx))
        jobs.append(proc)
        proc.start()
        while True:
            if canStartMoreJobs(jobs):
                break
            else:
                print "$$$$ WAITING FOR PROCESSOR SPACE $$$$"
                time.sleep(1)

    for proc in jobs:
        proc.join()

    print "#"*10+"Writing Results"+"#"*10
    writeAlignment(args.output, sVocab, tVocab, nearestVectorsIdx)


