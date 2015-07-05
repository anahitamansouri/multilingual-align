# -*- coding: utf-8 -*-
__author__ = 'Jasneet Sabharwal <jsabharw@sfu.ca>'

import argparse
import operator
from collections import defaultdict, Counter


def readPartialAlignments(alignmentFile):
    result = defaultdict(list)
    with open(alignmentFile) as inFile:
        for idx, line in enumerate(inFile):
            if idx%5000 == 0:
                print idx
            line = line.strip('\r\n')
            if line:
                lineSplit = line.split('\t')
                result[lineSplit[0].strip()].append(lineSplit[1].strip())
        print idx
    return result


def readPOSTaggedCorpora(corporaFile):
    resultVocabPOS = {}
    vocabCounts = defaultdict(int)
    vocabMultiPOS = defaultdict(list)

    with open(corporaFile) as inFile:
        for idx, line in enumerate(inFile):
            if idx%5000 == 0:
                print idx
            line = line.strip('\r\n')
            if line:
                lineSplit = line.split(' ')
                for token in lineSplit:
                    tokenPOSSplit = token.split('_', 1)
                    pos = None
                    if tokenPOSSplit[0] == '' and tokenPOSSplit[1].startswith('_'):
                        token = tokenPOSSplit[1][:-2]
                        pos = tokenPOSSplit[1][-2:]
                    elif not(tokenPOSSplit[0] == ''):
                        token = tokenPOSSplit[0]
                        pos = tokenPOSSplit[1]
                    elif tokenPOSSplit[0] == '' and not(tokenPOSSplit[1].startswith('_')) and token.startswith('_'):
                        token = '_'
                        pos = tokenPOSSplit[1]

                    if pos.startswith('NN'):
                        pos = 'NN'
                    elif pos.startswith('JJ'):
                        pos = 'JJ'
                    elif pos.startswith('RB'):
                        pos = 'RB'
                    elif pos.startswith('VB'):
                        pos = 'VB'
                    else:
                        pos = None

                    if pos is not None:
                        vocabMultiPOS[token.lower()].append(pos)
                        vocabCounts[token.lower()] += 1
        print idx

    for word, poss in vocabMultiPOS.iteritems():
        mostCommonPOS = Counter(poss).most_common(1)[0][0]
        resultVocabPOS[word] = mostCommonPOS

    sortedVocabCounts = sorted(vocabCounts.items(), key=operator.itemgetter(1), reverse=True)

    return resultVocabPOS, sortedVocabCounts


def getPOSFreq(pos, nFreq, vFreq, rbFreq, jjFreq):
    if pos == 'NN':
        return nFreq
    elif pos == 'VB':
        return vFreq
    elif pos == 'JJ':
        return jjFreq
    elif pos == 'RB':
        return rbFreq


def hasPOSCountLimitReached(posCounts, nFreq, vFreq, rbFreq, jjFreq):
    returnVal = True
    for pos, count in posCounts.iteritems():
        if count < getPOSFreq(pos, nFreq, vFreq, rbFreq, jjFreq):
            returnVal = False
    return returnVal


def filterVocabByFrequency(vocabPOS, vocabCounts, nFreq, vFreq, rbFreq, jjFreq):
    result = []
    posCounted = defaultdict(int)
    for word, count in vocabCounts:
        pos = vocabPOS[word]
        if posCounted[pos] < getPOSFreq(pos, nFreq, vFreq, rbFreq, jjFreq):
            result.append(word)
            posCounted[pos] += 1
        if hasPOSCountLimitReached(posCounted, nFreq, vFreq, rbFreq, jjFreq):
            print '#'*5 + 'POS COUNTS: {}'.format(' || '.join([pos+' - '+str(count) for pos, count in posCounted.iteritems()])) + '#'*5
            break
    return result


def writePartialAlignments(sourceWords, alignments, outputFile):
    with open(outputFile, 'w') as outFile:
        for sourceWord in sourceWords:
            targetWords = alignments[sourceWord]
            for targetWord in targetWords:
                outFile.write('{}\t{}\n'.format(sourceWord, targetWord))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--alignments', dest='alignmentFile', help='Partial alignments file')
    parser.add_argument('-i', '--input', dest='posFile', help='POS tagged corpora')
    parser.add_argument('-n', '--nouns', dest='nounFreq', type=int, default=250,
                        help='Noun frequency (default: 250)')
    parser.add_argument('-v', '--verbs', dest='verbFreq', type=int, default=250,
                        help='Verb frequency (default: 250)')
    parser.add_argument('-rb', '--adverbs', dest='advFreq', type=int, default=250,
                        help='Adverb frequency (default: 250)')
    parser.add_argument('-jj', '--adjectives', dest='adjFreq', type=int, default=250,
                        help='Adjective frequency (default: 250)')
    parser.add_argument('-o', '--output', dest='outputFile', help='Filtered partial alignment file')
    args = parser.parse_args()

    print '$'*10 + ' READING PARTIAL ALIGNMENTS ' + '$'*10
    partialAlignments = readPartialAlignments(args.alignmentFile)

    print '$'*10 + ' READING POS TAGGED CORPORA ' + '$'*10
    vocabPOS, vocabCount = readPOSTaggedCorpora(args.posFile)

    print '$'*10 + ' FILTERING VOCAB BY FREQUENCY ' + '$'*10
    filteredWords = filterVocabByFrequency(vocabPOS, vocabCount, args.nounFreq, args.verbFreq, args.advFreq,
                                           args.adjFreq)

    print '$'*10 + ' WRITING PARTIAL ALIGNMENTS ' + '$'*10
    writePartialAlignments(filteredWords, partialAlignments, args.outputFile)