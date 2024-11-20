#ifndef PLM_ALPHABET_H
#define PLM_ALPHABET_H

#include <vector>
#include <string>
#include <map>
#include "../matrix.h"

/*
    {
        '<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3,
        'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10,
        'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16,
        'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22,
        'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28,
        '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32
    }
 */

const int CLS_TOKEN = 0;
const int EOS_TOKEN = 2;
const int UNK_TOKEN = 3;

const int ALPHABET_SIZE = 33;

const std::map<char, int> AA_ONE_TO_ESM = {
    {'L', 4}, {'A', 5}, {'G', 6}, {'V', 7}, {'S', 8}, {'E', 9}, {'R', 10},
    {'T', 11}, {'I', 12}, {'D', 13}, {'P', 14}, {'K', 15}, {'Q', 16},
    {'N', 17}, {'F', 18}, {'Y', 19}, {'M', 20}, {'H', 21}, {'W', 22},
    {'C', 23},
};

const std::map<char, int> AA_ONE_TO_FOLDING = {
    {'A', 0}, {'R', 1}, {'N', 2}, {'D', 3}, {'C', 4},
    {'Q', 5}, {'E', 6}, {'G', 7}, {'H', 8}, {'I', 9},
    {'L', 10}, {'K', 11}, {'M', 12}, {'F', 13}, {'P', 14},
    {'S', 15}, {'T', 16}, {'W', 17}, {'Y', 18}, {'V', 19},
    {'X', 20}
};

matrix<int> tokenize_esm_aatype(const std::string &seq) {
    std::vector<int> tokens;
    for (char c : seq) {
        if (AA_ONE_TO_ESM.find(c) != AA_ONE_TO_ESM.end()) {
            tokens.push_back(AA_ONE_TO_ESM.at(c));
        } else {
            tokens.push_back(UNK_TOKEN);
        }
    }
    matrix<int> result(tokens.size(), 1);
    for (int i = 0; i < tokens.size(); i ++) {
        *result(i, 0) = tokens[i];
    }
    return result;
}

matrix<int> tokenize_folding_aatype(const std::string &seq) {
    std::vector<int> tokens;
    for (char c : seq) {
        if (AA_ONE_TO_FOLDING.find(c) != AA_ONE_TO_FOLDING.end()) {
            tokens.push_back(AA_ONE_TO_FOLDING.at(c));
        } else {
            tokens.push_back(20);
        }
    }
    matrix<int> result(tokens.size(), 1);
    for (int i = 0; i < tokens.size(); i ++) {
        *result(i, 0) = tokens[i];
    }
    return result;
}


matrix<int> make_residx(int seqlen) {
    matrix<int> residx(seqlen, 1);
    for (int i = 0; i < seqlen; i++) {
        *residx(i, 0) = i;
    }
    return residx;
}

#endif // PLM_ALPHABET_H