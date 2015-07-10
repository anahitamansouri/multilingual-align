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
            nearestNeighbours = nearestVectorsIdx[idx]
            for neighbour in nearestNeighbours:
                tWord = tVocab[neighbour]
                outFile.write('{}\t{}\n'.format(sWord, tWord))


def findClosestVector(sIdx, sVector, tVectors, returnDict, neighbours):
    print 'FIND CLOSEST VECTOR - {} - pid: {}'.format(sIdx, getpid())

    cosineDistances = []
    for tVector in tVectors:
        cosineDistance = cosine(sVector, tVector)
        cosineDistances.append(cosineDistance)
    cosineDistances = np.array(cosineDistances)
    neighbourIndices = sorted(np.argpartition(cosineDistances, neighbours)[:neighbours])
    returnDict[sIdx] = neighbourIndices


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
    optParser.add_argument("-k", "--neighbours", dest="neighbours", type=int, default=1,
                           help="Number of nearest neighbours (default=1)")
    optParser.add_argument("-o", "--output", dest="output", help="Output alignment file")
    args = optParser.parse_args()

    print 'SOURCE VECTORS: {}\nTARGET VECTORS: {}\nNEIGHBOURS: {}\nOUTPUT: {}'.format(args.sourceVectors,
                                                                                      args.targetVectors,
                                                                                      args.neighbours, args.output)

    print "#"*10+"Reading Vectors"+"#"*10
    sVectors, sVocab = readVectors(args.sourceVectors)
    tVectors, tVocab = readVectors(args.targetVectors)

    print "#"*10+"Calculating Cosines"+"#"*10
    manager = multiprocessing.Manager()
    nearestVectorsIdx = manager.dict()
    jobs = []

    for sIdx, sVector in enumerate(sVectors):
        print '\tCalculating Cosine for: {} - {}'.format(sIdx, sVocab[sIdx])

        proc = multiprocessing.Process(target=findClosestVector, args=(sIdx, sVector, tVectors, nearestVectorsIdx,
                                                                       args.neighbours))
        jobs.append(proc)
        proc.start()
        while True:
            if canStartMoreJobs(jobs):
                break
            else:
                print "$$$$ WAITING FOR PROCESSOR SPACE $$$$"
                time.sleep(0.1)

    for proc in jobs:
        proc.join()

    print "#"*10+"Writing Results"+"#"*10
    writeAlignment(args.output, sVocab, tVocab, nearestVectorsIdx)


