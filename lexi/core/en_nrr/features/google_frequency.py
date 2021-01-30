# -*- coding: utf-8 -*-

import numpy as np


class GoogleFrequency:
    def __init__(self, google_frequency_file):
        google_frequencies = {}
        for line in open(google_frequency_file, encoding='utf-8'):
            line_tokens = [t.strip() for t in line.strip().split('\t')]
            try:
                count = int(line_tokens[1])
                if count > 100000:
                    google_frequencies[line_tokens[0]] = np.log10(count)
            except IndexError:
                print("Error: the following has no corresponding word: " + str(line_tokens))
                pass
        self.google_frequencies = google_frequencies

    def get_feature(self, phrase):
        phrase = phrase.lower()
        if phrase in self.google_frequencies:
            return self.google_frequencies[phrase]
        return 0
