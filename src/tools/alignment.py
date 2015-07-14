__author__ = 'Jasneet Sabharwal <jsabharw@sfu.ca>'

# This file contains extra methods needed for alignments

import sys


def alignmentFromIBM1(bitext, tProb, alignmentFile, numLines=None):
    """
    For each sentence pair, find the alignment for each source word and write to file
    :param bitext: the parallel sentences
    :param tProb: translation probabilities
    :param alignmentFile: file to which alignments should be written
    :param numLines: number of lines for which alignments should be written and printed
    :return: Alignments for all sentence pairs
    """
    with open(alignmentFile, 'w') as outFile:
        allAlignments = []
        for (n, (source, target)) in enumerate(bitext):
            alignments = set()
            for sWordIdx, sWord in enumerate(source):
                maxProb = 0
                argmax = -1

                for tWordIdx, tWord in enumerate(target):
                    if tProb[(sWord, tWord)] > maxProb:
                        maxProb = tProb[(sWord, tWord)]
                        argmax = tWordIdx

                alignments.add((sWordIdx, argmax))

            allAlignments.append(alignments)

            for (sWordIdx, argmax) in alignments:
                #sys.stdout.write('%i-%i ' % (sWordIdx+1, argmax+1))
                outFile.write('%i-%i ' % (sWordIdx+1, argmax+1))
            #sys.stdout.write('\n')
            outFile.write('\n')
            if numLines and (n == numLines - 1):
                break
        return allAlignments


def alignmentFromIBM2(bitext, tProb, qProb, alignmentFile, numLines=None):
    """
    For each sentence pair, find the alignment for each source word and target word, and write to file
    :param bitext: the parallel sentences
    :param tProb: translation probabilities
    :param qProb: alignment probabilities
    :param alignmentFile: file to which alignments should be written
    :param numLines: number of lines for which alignments should be written and printed
    :return: Alignments for all sentence pairs
    """
    with open(alignmentFile, 'w') as outFile:
        allAlignments = []
        for n, (source, target) in enumerate(bitext):
            alignments = set()

            for j, sWord in enumerate(source):
                maxProb = 0.0
                argmax = -1

                for i, tWord in enumerate(target):
                    prob = tProb[(sWord, tWord)] * qProb[(j, i, len(target), len(source))]
                    if prob > maxProb:
                        maxProb = prob
                        argmax = i
                alignments.add((j, argmax))

            allAlignments.append(alignments)

            for (sWordIdx, argmax) in alignments:
                outFile.write('%i-%i ' % (sWordIdx+1, argmax+1))
            outFile.write('\n')

            if numLines and (n == numLines - 1):
                break

    return allAlignments


def intersectAlignmentModels(bitext, probSourceTarget, probTargetSource, alignmentFile=None, numLines=None):
    """

    :param bitext: the parallel sentences
    :param probSourceTarget: translation probabilities for source|target
    :param probTargetSource: translation probabilities for target|source
    :param alignmentFile: file to which alignments should be written
    :param numLines: number of lines for which alignments should be written and printed
    :return: Alignments for all sentence pairs
    """
    allAlignments = []

    if alignmentFile:
        outFile = open(alignmentFile, 'w')

    for (n, (source, target)) in enumerate(bitext):
        sourceTargetAlignments = set()
        targetSourceAlignments = set()

        for sWordIdx, sWord in enumerate(source):
            maxProb = 0
            argmax = -1

            for tWordIdx, tWord in enumerate(target):
                if probSourceTarget[(sWord, tWord)] > maxProb:
                    maxProb = probSourceTarget[(sWord, tWord)]
                    argmax = tWordIdx
            sourceTargetAlignments.add((sWordIdx, argmax))

        for tWordIdx, tWord in enumerate(target):
            maxProb = 0
            argmax = -1

            for sWordIdx, sWord in enumerate(source):
                if probTargetSource[(tWord, sWord)] > maxProb:
                    maxProb = probTargetSource[(tWord, sWord)]
                    argmax = sWordIdx
            targetSourceAlignments.add(argmax, tWordIdx)

        # TODO: intersecting after getting tsAlignments shouldn't change anything, should be better
        intersect = sourceTargetAlignments.intersection(targetSourceAlignments)
        allAlignments.append(intersect)

        for sWordIdx, argmax in intersect:
            if alignmentFile:
                # TODO: shouldn't it be sWordIdx+1, argmax+1
                outFile.write('%i-%i ' % (sWordIdx+1, argmax+1))
            else:
                sys.stdout.write('%i-%i'  % (sWordIdx+1, argmax+1))
        if alignmentFile:
            outFile.write('\n')
        else:
            sys.stdout.write('\n')

        if numLines and (n == numLines - 1):
            break

    if alignmentFile:
        outFile.close()

    return allAlignments

