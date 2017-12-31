#!/usr/bin/env python
#
# Program to perform Optical Character Recognition
# CS B551 Fall 2017, Assignment #3
#
# Your names and user ids:
# Saurabh Agrawal  -  agrasaur
# Gaurav Derasaria -  gderasar
# Sharad Ghule     -  ssghule
#
# How we are representing this problem?
# A) We have mapped this problem into a Hidden Markov model. We consider each test letter as a new observed state.
#    And we try to compute the hidden states (most likely character for the observed state).
#
# Hidden Markov model requires calculation of three probability tables:
# i) Initial Probabilities: We compute these probabilities from the training text file. These are the probabilities
#  of a character being the first letter of a statement. For computing initial probabilities, we traversed the
# training file and counted the first characters of the sentence, and then normalized these probabilities.
#
# ii) Transition probabilities: This probability table stores the probability of transitioning from one character to
#  another. For computing this probability table, we counted the occurrences of a character after another character.
#
# iii) Emission probabilities: These are the probabilities of the test character grid representing a particular
#  character. We used a naive bayes approach to compute the emission probabilities of a test character. The grid
#  of the test character was matched against the grid of the train letters. The probability of pixel being noisy was
#  the total unmatched pixels by total pixels and the probability of pixel not being noisy was the total matched pixel
#  by the total number of pixels. If the pixels matched, probabilities of the pixel would be the probability of
#  pixel not being noisy and if the pixels did not match the probability of the pixel would be the probability of a
#  noisy pixel. The emission probability was computed by taking the product of all pixel probabilities.
#
# How we did OCR?
# i) Simplified: The algorithm took into account only the emission and character probabilities. There was no
#    connection between the two hidden states. The character assigned to each noisy character was simply decided by
#    the emission probability and the probability of the character occurring.
# ii) Variable Elimination: The algorithm all three probabilities. At every transition we would compute the sum of
#    all the probabilities from the previous character times the transition from previous character to the current
#    character. Then we would multiply the sum with the emission probability for that character. Finally, we assign the
#    computed value to the character. The train character with the highest probability would be the assigned character.
# iii) Viterbi Algorithm: Again, the algorithm considers all three probabilities. The Viterbi algorithm would store
#    the most likely previous char and the probability of transitioning from the previous character to the current
#    character times the emission probability of the current character. When the complete table was populated we
#    would backtrack from the most likely state at the end always going to the previous state at the index stored
#    on the current character.

from collections import defaultdict
import math
import sys

# Set to store all the possible characters
TRAIN_LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' ")
CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25
max_val = 10000000.0  # Maximum value
min_val = sys.float_info.epsilon  # Smallest possible value


class OCRSolver:
    """Class that contains all the logic to perform Optical Character Recognition (OCR) for test image 
    """

    def __init__(self, train_letters, test_letters, train_txt_fname):
        """
        :param train_letters:   Dictionary that maps from a character to the correct representation of the character
        :param test_letters:    List of test character representations for which we need to find the correct character
        :param train_txt_fname: Name of the text file that will be used to compute non-emission probabilities
        """
        # Train letters is a dictionary that maps a character to the correct representation of the character
        # This representation is a list of 25 strings of length 14
        # For example: {'a': ['', '', '', ...], 'b': ['', '', '', ...] ...}
        self.train_letters = train_letters

        # List of test character representations for which we need to find the correct character
        # This representation is a list of 25 strings of length 14
        # For example: [['', '', '', ...], ['', '', '', ...] ...]
        self.test_letters = test_letters

        # Dictionary that represents the initial probability table
        # Stores the negative logs of probabilities of starting with a particular character
        # For example: {'a': 1.234, 'b': 2.234 ...}
        self.init_prob = dict()

        # Dictionary that stores the overall probabilities of characters
        # Stores the negative logs of probabilities of overall characters
        # For example: {'a': 1.234, 'b': 2.234 ...}
        self.char_prob = dict()
        for char in TRAIN_LETTERS:
            self.char_prob[char] = min_val

        # Dictionary of dictionaries that represent the transition probability table
        # Stores the negative logs of probabilities of transitioning from one character to another
        # For example: {'a': {'a': 1.234, 'b': 2.234 ..}, 'b': {'a': 1.234, 'b': 2.234} ...}
        self.trans_prob = defaultdict(dict)
        for row_char in TRAIN_LETTERS:
            for col_char in TRAIN_LETTERS:
                self.trans_prob[row_char][col_char] = min_val

        # Dictionary of dictionaries that represent the emission probability table
        # Stores the negative logs of probabilities of noisy character matching with each character in TRAIN_LETTERS
        # For example: { 1: {'a': 1.234, 'b': 2.234 ..}, 2: {'a': 1.234} ...}
        self.emit_prob = defaultdict(dict)

        self.train(train_txt_fname)

    def print_inputs(self):
        def print_dict(dict_to_print, items_to_print):
            """Static method to print the dictionary
            :param dict_to_print:  Dictionary that needs to be printed
            :param items_to_print: Number of items that need to be printed
            """
            for key, val in dict_to_print.iteritems():
                items_to_print -= 1
                if items_to_print == 0:
                    break
                print 'Key ->', key, '   Val ->', val

        # print_inputs() starts from here
        print 'Printing initial probabilities', len(self.init_prob)
        print 'Size of initial probabilities', sys.getsizeof(self.init_prob)
        print_dict(self.init_prob, sys.maxint)

        print 'Printing tag probabilities', len(self.char_prob)
        print 'Size of tag probabilities', sys.getsizeof(self.init_prob)
        print_dict(self.char_prob, sys.maxint)

        print 'Printing transition probabilities', len(self.trans_prob)
        print 'Size of transition probabilities', sys.getsizeof(self.trans_prob)
        print_dict(self.trans_prob, sys.maxint)

        print 'Printing emission probabilities', len(self.emit_prob)
        print 'Size of emission probabilities', sys.getsizeof(self.emit_prob)
        print_dict(self.emit_prob, 50)
        # sys.exit(0)

    @staticmethod
    def normalize_dict(dict_to_normalize):
        """Transforms count of a dictionaries to natural log of the probabilties
        :param dict_to_normalize: Dictionary that needs to be normalized
        :return:
        """
        total_log = math.log(sum(dict_to_normalize.values()))
        for key, val in dict_to_normalize.iteritems():
            dict_to_normalize[key] = max_val if val < 1 else total_log - math.log(val)

    def compute_emission(self):
        """Populates the emission probabilities table (self.emit_prob) by comparing the test letters with training
           characters (self.train_letters) and using a Naive Bayes classifier to compute the matching probability
        """

        def match_grids(grid1, grid2):
            """Performs the cell-wise character matching of 2 grids (list of strings) and returns the number of matches
               Example: For input strings: ['*','_','*'] and ['*','*','*'],
               this method will return 8
               Constraint: The size of the 2 grids should be equal
            :param grid1: First grid (list of strings)
            :param grid2: Second grid (list of strings)
            :return: The number of cell-wise character matches
            """
            matches = 0
            for row1, row2 in zip(grid1, grid2):
                for ch1, ch2 in zip(row1, row2):
                    if ch1 == ch2:
                        matches += 1
            return matches

        # compute_emission() starts from here
        total_pixels = CHARACTER_WIDTH * CHARACTER_HEIGHT

        for curr_index, test_letter in enumerate(self.test_letters):
            for train_letter, train_letter_grid in self.train_letters.iteritems():
                # Compute the probability of a noisy character representing a char from the
                matched = match_grids(test_letter, train_letter_grid)
                unmmatched = total_pixels - matched
                match_prob = (matched + 0.0) / total_pixels

                # Computing the probability of the test character being the train character
                prob = (match_prob ** matched) * ((1 - match_prob) ** unmmatched)
                # self.emit_prob[curr_index][train_letter] = total_pixels_log - math.log(matched + 0.00001) - \

                self.emit_prob[curr_index][train_letter] = max_val if prob == 0 else -math.log(prob)
                # print(self.emit_prob)

    def train(self, train_txt_fname):
        """Calculates the three probability tables for the given data
        :param train_txt_fname: Filename of the file containing the text corpus
        """

        def clean_string(str_to_clean):
            """Cleans the given string by removing special characters
            :param str_to_clean: The string that needs to be cleaned
            :return: The clean string
            """
            str_to_clean = list(str_to_clean)
            idx = 0
            while idx < len(str_to_clean) - 1:
                curr_ch = str_to_clean[idx]
                next_ch = str_to_clean[idx + 1]
                if curr_ch not in TRAIN_LETTERS:
                    str_to_clean[idx] = ' '
                if next_ch not in TRAIN_LETTERS:
                    str_to_clean[idx + 1] = ' '
                if next_ch == ' ' and (curr_ch == '.' or curr_ch == ' '):
                    del str_to_clean[idx + 1]
                else:
                    idx += 1
            return str_to_clean

        # train() starts from here
        with open(train_txt_fname, 'r') as train_txt_file:
            train_text = clean_string(train_txt_file.read())
            is_initial_letter = True
            for index in range(0, len(train_text) - 1):
                curr_char = train_text[index]
                next_char = train_text[index + 1]

                if is_initial_letter:
                    if curr_char not in self.init_prob:
                        self.init_prob[curr_char] = 0
                    self.init_prob[curr_char] += 1
                    is_initial_letter = False

                if curr_char == '.':
                    is_initial_letter = True

                self.trans_prob[curr_char][next_char] += 1
                self.char_prob[curr_char] += 1

        # Normalizing initial probabilities table
        self.normalize_dict(self.init_prob)

        # Normalizing tag probabilities table
        self.normalize_dict(self.char_prob)

        # Normalizing transition probabilities table
        for row_dict in self.trans_prob.values():
            # total_log = math.log(sum(row_dict.values()))
            # for key, val in row_dict.iteritems():
            #     row_dict[key] = 10000 if val < 1 else total_log - math.log(val)
            self.normalize_dict(row_dict)

        self.compute_emission()
        # self.print_inputs()

    def get_emission_probs(self, noisy_char):
        """Computes emission probabilities for a word
        :param noisy_char: string word
        :return: A dictionary mapping tag to the emission probability {'noun': 1.234, 'verb': 2.345) ...)
        """
        emission_prob_dict = dict()
        for char in TRAIN_LETTERS:
            if noisy_char in self.emit_prob and char in self.emit_prob[noisy_char]:
                emission_prob_dict[char] = self.emit_prob[noisy_char][char]
            else:
                emission_prob_dict[char] = max_val
        return emission_prob_dict

    def simplified(self):
        """Returns the best matching character for every test characters
        :return: set of characters (string) that best match the test characters
        """
        output_chars = list()
        for index, test_letter_grid in enumerate(self.test_letters):
            # print 'Printing test letter'
            # self.print_char_grid(test_letter_grid)

            (best_ch, best_prob) = (None, sys.float_info.max)
            for ch, prob in self.emit_prob[index].iteritems():
                curr_prob = prob + self.char_prob[ch]
                if curr_prob < best_prob:
                    (best_ch, best_prob) = (ch, curr_prob)
            output_chars.append(best_ch)
            # print 'Printing train letter'
            # self.print_char_grid(self.train_letters[best_ch])
        print 'Simple:', ''.join(output_chars)

    def hmm_ve(self):
        """Performs OCR by performing variable elimination on Hidden Markov model constructed by keeping
            the test characters on observed states, and computing the best likely values of hidden states
        :return: set of characters (string) that best match the test characters
        """
        chars = dict()  # Dictionary to store all probabilities
        char_dict = dict()
        for char in TRAIN_LETTERS:  # Calculating for the first character
            if char not in self.emit_prob[0].keys() or char not in self.init_prob.keys():
                char_dict[char] = sys.float_info.max
            else:
                char_dict[char] = self.char_prob[char] + self.emit_prob[0][char]
        chars[0] = char_dict

        for w in range(1, len(self.test_letters)):  # Calculating for the remaining characters
            char_dict = dict()
            prev_chars_dict = chars[w - 1]
            for char in TRAIN_LETTERS:
                prob = (1 / sys.float_info.max)
                for prev_char in prev_chars_dict:  # Scanning probabilities of the previous character
                    current_prob = prev_chars_dict[prev_char] + self.trans_prob[prev_char][char]
                    prob += math.exp(-current_prob)  # For adding all the probabilities
                if char not in self.emit_prob[w].keys():
                    char_dict[char] = sys.float_info.max  # Assign maximum value for a new character
                else:
                    char_dict[char] = -math.log(prob) + self.emit_prob[w][char]
            chars[w] = char_dict
        print 'HMM VE:', ''.join(
            [chars[w].keys()[chars[w].values().index(min(chars[w].values()))] for w in
             range(len(self.test_letters))])  # Printing the characters that have maximum probability

    def hmm_viterbi(self):
        """Performs OCR by solving Viterbi algorithm on Hidden Markov model constructed by keeping
           the test characters on observed states, and computing the best likely values of hidden states
        :return: set of characters (string) that best match the test characters
        """
        char_list = list(TRAIN_LETTERS)  # Converting tag_set to a list to have indexes to refer
        rows = len(char_list)
        cols = len(self.test_letters)
        vit_matrix = [[None] * cols for i in range(rows)]

        # Storing a tuple in each cell (index of the previous cell, probability of the current cell)
        for col_index in range(len(self.test_letters)):
            curr_emission_probs = self.get_emission_probs(col_index)

            for row_index, curr_char in enumerate(char_list):
                # Computing the probabilities for the first column
                if col_index == 0:
                    init_prob = self.init_prob[curr_char] if curr_char in self.init_prob else max_val
                    vit_matrix[row_index][col_index] = (-1, curr_emission_probs[curr_char] + init_prob)
                # Computing the probabilities of the other columns
                else:
                    best_prob_tuple = (-1, 200000000.0)
                    for prev_row_index, prev_char in enumerate(char_list):
                        prev_prob = vit_matrix[prev_row_index][col_index - 1][1]
                        curr_prob = prev_prob + self.trans_prob[prev_char][curr_char] + curr_emission_probs[curr_char]
                        if curr_prob < best_prob_tuple[1]:
                            best_prob_tuple = (prev_row_index, curr_prob)
                    vit_matrix[row_index][col_index] = (best_prob_tuple[0], best_prob_tuple[1])

        # Backtracking to fetch the best path
        # Finding the cell with the max probability from the last column
        (max_index, max_prob) = (-1, max_val)
        for row in range(rows):
            curr_prob = vit_matrix[row][cols - 1][1]
            if curr_prob < max_prob:
                (max_index, max_prob) = (row, curr_prob)

        output_list = list()  # List to store the output tags
        # Adding the best path to output list
        for col in range(cols - 1, 0, -1):
            output_list.insert(0, char_list[max_index])
            max_index = vit_matrix[max_index][col][0]
        output_list.insert(0, char_list[max_index])
        print 'HMM MAP:', ''.join(output_list)
