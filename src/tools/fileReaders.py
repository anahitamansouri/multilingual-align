# -*- coding: utf-8 -*-
__author__ = 'Jasneet Sabharwal <jsabharw@sfu.ca>'


from collections import defaultdict


def readPartialAlignments(annotationFile, sourceColumn, targetColumn):
    """
    Reads the partial annotations from tab separated file
    :param annotationFile: Annotation file
    :param sourceColumn: Column number of source words
    :param targetColumn: Column number of target words
    :return:
    """
    partialAlignments = defaultdict(list)
    with open(annotationFile) as inFile:
        for line in inFile:
            line = line.strip('\r\n')
            if line:
                lineSplit = line.split()
                sWord = lineSplit[sourceColumn].strip()
                tWord = lineSplit[targetColumn].strip()
                partialAlignments[sWord].append(tWord)
    return partialAlignments
