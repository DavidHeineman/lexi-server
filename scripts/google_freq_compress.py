import numpy as np

# Loads a large n-gram file and saves all words with counts above CUTOFF
large_file = 'res/en/nrr/google_freq_bigram_all.bin'
CUTOFF = 10000

total = 0
nextone = 0
orig = 0
google_frequencies = {}
for line in open(large_file, encoding='utf-8'):
    line_tokens = [t.strip() for t in line.strip().split('\t')]
    try:
        count = int(line_tokens[1])
        if count > CUTOFF:
            google_frequencies[line_tokens[0]] = count
            total += 1
            nextone = 0
        else:
            orig += 1
    except IndexError:
        print("Error: the following has no corresponding word: " + str(line_tokens))
        pass
    if (total % 1000000 == 0 and nextone == 0):
        nextone = 1
        print("N-gram count: " + str(total))
print('Total n-grams saved: %d of %d | %.5f%s' % (total, (total + orig), (total / orig) * 100, "%"))

import csv
with open('res/en/nrr/google_freq_all.bin', 'w', encoding='UTF-8') as f:
    for key in google_frequencies.keys():
        f.write("%s\t%s\n"%(key, google_frequencies[key]))