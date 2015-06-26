#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'amansour'

import sys

def grade_align(f_data, e_data, reference, system, output):
    (size_a, size_s, size_a_and_s, size_a_and_p) = (0.0,0.0,0.0,0.0)
    for (n, (f, e, g, a)) in enumerate(zip(open(f_data), open(e_data), open(reference), open(system))):
      print n
      fwords = f.strip().split()
      ewords = e.strip().split()

      # check

      size_f = len(fwords)
      size_e = len(ewords)
      try:
        alignment = set([tuple(map(int, x.split("-"))) for x in a.strip().split()])
        for (i,j) in alignment:
          if (i>size_f or j>size_e):
            #print 'i',i,'j',j,'size_F',size_f,'size_e',size_e,'\n'
            sys.stderr.write("WARNING (%s): Sentence %d, point (%d,%d) is not a valid link\n" % (sys.argv[0],n,i,j))
          pass
      except (Exception):
        sys.stderr.write("ERROR (%s) line %d is not formatted correctly:\n  %s" % (sys.argv[0],n,a))
        sys.stderr.write("Lines can contain only tokens \"i-j\", where i and j are integer indexes into the French and English sentences, respectively.\n")
        sys.exit(1)

      # grade

      sure = set([tuple(map(int, x.split("-"))) for x in filter(lambda x: x.find("-") > -1, g.strip().split())])
      possible = set([tuple(map(int, x.split("?"))) for x in filter(lambda x: x.find("?") > -1, g.strip().split())])
      alignment = set([tuple(map(int, x.split("-"))) for x in a.strip().split()])
      size_a += len(alignment)
      size_s += len(sure)
      size_a_and_s += len(alignment & sure)
      size_a_and_p += len(alignment & possible) + len(alignment & sure)
#      if (i<opts.num_sents):
#        output.write("  Alignment %i  KEY: ( ) = guessed, * = sure, ? = possible\n" % i)
#        output.write("  ")
#        for j in ewords:
#          output.write("---")
#        output.write("\n")
#        for (i, f_i) in enumerate(fwords):
#          output.write(" |")
#          for (j, _) in enumerate(ewords):
#            (left,right) = ("(",")") if (i,j) in alignment else (" "," ")
#            point = "*" if (i,j) in sure else "?" if (i,j) in possible else " "
#            output.write("%s%s%s" % (left,point,right))
#          output.write(" | %s\n" % f_i)
#        output.write("  ")
#        for j in ewords:
#          output.write("---")
#        output.write("\n")
#        for k in range(max(map(len, ewords))):
#          output.write("  ")
#          for word in ewords:
#            letter = word[k] if len(word) > k else " "
#            output.write(" %s " % letter)
#          output.write("\n")
#        output.write("\n")

    precision = size_a_and_p / size_a
    recall = size_a_and_s / size_s
    aer = 1 - ((size_a_and_s + size_a_and_p) / (size_a + size_s))
    output.write("Precision = %f\nRecall = %f\nAER = %f\n" % (precision, recall, aer))
    print >>sys.stderr, "Precision = %f\nRecall = %f\nAER = %f\n" % (precision, recall, aer)
    return True

def convert_giza_out_to_aer_out(readFile, writeFile):
    alignment = open(readFile,'r')
    alignment_aer_format = open(writeFile,'w')
    for line in alignment:
        #set1 = set()
        line = line.strip()
        l = line.split(' ')
        #print line
        for pair in l:
	    if '-' in pair:
            	[a,b] = pair.split('-')
		alignment_aer_format.write("%i-%i " % (int(a)+1, int(b)+1))
	    elif '?' in pair:
		[a,b] = pair.split('?')
		#print a, '?', b
	        alignment_aer_format.write("%i?%i " % (int(a)+1, int(b)+1))
            #alignment_aer_format.write(int(a)+1 + "-" + int(b)+1 + " ")
        alignment_aer_format.write("\n")
    alignment.close()
    alignment_aer_format.close()
