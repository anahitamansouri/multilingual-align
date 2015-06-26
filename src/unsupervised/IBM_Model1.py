#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'amansour'

from collections import defaultdict

def EM_IBM1(source_count, target_count, st_count,bitext):
    c = defaultdict(int)
    total = defaultdict(int)
    t = defaultdict(int)

    #initialize t uniformly
    for (source,target) in st_count:
        t[(source,target)] = 1.0/len(source_count)
    for cnt in range(10):
        c = defaultdict(int)
        total = defaultdict(int)

        for (S,D) in bitext:
            for s_i in S:
                Z = 0 #normalizer
                for d_j in D:
                    Z += t[(s_i,d_j)]
                for d_j in D:
                    c[(s_i,d_j)] += t[(s_i,d_j)]/Z
                    total[d_j] += t[(s_i,d_j)]/Z
        for (s,d) in st_count:
            t[(s,d)] = c[(s,d)]/total[d]
    return t


