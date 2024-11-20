#ifndef PLM_ALPHABET_H
#define PLM_ALPHABET_H

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

#endif // PLM_ALPHABET_H