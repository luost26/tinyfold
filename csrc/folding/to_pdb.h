#ifndef FOLDING_TO_PDB_H
#define FOLDING_TO_PDB_H

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include "../matrix.h"
#include "ipa.h"
#include "geometric.h"


const char *aa_index_to_three[21] = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "UNK"
};

const char *aa_index_to_one[21] {
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
    "X"
};

const char *pdb_atom_format = "ATOM  %5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f           %c\n";


struct AminoAcid {
    char const *aaname;
    float pos_N[3] = {0.0, 0.0, 0.0};
    float pos_CA[3] = {0.0, 0.0, 0.0};
    float pos_C[3] = {0.0, 0.0, 0.0};
    float pos_O[3] = {0.0, 0.0, 0.0};

    AminoAcid(const char * aaname) {
        this->aaname = aaname;
    }

    std::string to_pdb(int &count_atoms, int resseq) {
        char buf[1024];
        int loc = 0;
        loc += sprintf(buf + loc, pdb_atom_format, ++count_atoms, " N  ", aaname, "A", resseq, pos_N[0], pos_N[1], pos_N[2], 1.0, 0.0, 'N');
        loc += sprintf(buf + loc, pdb_atom_format, ++count_atoms, " CA ", aaname, "A", resseq, pos_CA[0], pos_CA[1], pos_CA[2], 1.0, 0.0, 'C');
        loc += sprintf(buf + loc, pdb_atom_format, ++count_atoms, " C  ", aaname, "A", resseq, pos_C[0], pos_C[1], pos_C[2], 1.0, 0.0, 'C');
        loc += sprintf(buf + loc, pdb_atom_format, ++count_atoms, " O  ", aaname, "A", resseq, pos_O[0], pos_O[1], pos_O[2], 1.0, 0.0, 'O');
        return std::string(buf);
    }
};

const float crd_N[3] = {-0.525, 1.363, 0.0};
const float crd_CA[3] = {0.0, 0.0, 0.0};
const float crd_C[3] = {1.526, 0.0, 0.0};
const float crd_O[3] = {2.153, -1.062, 0.0};

std::string to_pdb(const matrix<float> &r, const matrix<int> aatype) {
    int seqlen = r.n_rows;
    std::string pdb = "";

    std::vector<AminoAcid> amino_acids;
    for (int i = 0; i < seqlen; i ++) {
        int aat = *aatype(i, 0);
        if (aat < 0 || aat > 20) {
            aat = 20;
        }
        amino_acids.push_back(AminoAcid(aa_index_to_three[aat]));
        AminoAcid &aa = amino_acids.back();
        apply_affine(r(i, 0), crd_N, aa.pos_N);
        apply_affine(r(i, 0), crd_CA, aa.pos_CA);
        apply_affine(r(i, 0), crd_C, aa.pos_C);
        apply_affine(r(i, 0), crd_O, aa.pos_O);
    }

    for (int i = 0; i < seqlen - 1; i ++) {
        AminoAcid &aa_this = amino_acids[i];
        AminoAcid &aa_next = amino_acids[i + 1];
        float psi = dihedral_from_four_points(aa_this.pos_N, aa_this.pos_CA, aa_this.pos_C, aa_next.pos_N);
        float sin_psi = sinf(psi), cos_psi = cosf(psi);
        float affine_psi[16] = {
            1, 0, 0, 0,
            0, cos_psi, -sin_psi, 0,
            0, sin_psi, cos_psi, 0,
            0, 0, 0, 1
        };
        compose_affine(r(i, 0), affine_psi, affine_psi);
        apply_affine(affine_psi, crd_O, aa_this.pos_O);
    }

    std::string pdb_out = "";
    int count_atoms = 0;
    for (int i = 0; i < seqlen; i ++) {
        pdb_out += amino_acids[i].to_pdb(count_atoms, i + 1);
    }
    return pdb_out;
}


#endif // FOLDING_TO_PDB_H