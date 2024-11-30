import pickle
from pathlib import Path
from tinyfold.models.openfold.residue_constants import restype_3to1

import torch
import click


def load_pdb(pdb_path: Path):
    with open(pdb_path, "r") as f:
        lines = f.readlines()

    seq = []
    CA = []
    CB = []
    for line in lines:
        if line.startswith("ATOM") and line[13:15] == "CA":
            seq.append(restype_3to1[line[17:20]])
            CA.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            # Glycine
            if line[17:20] == "GLY":
                CB.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        elif line.startswith("ATOM") and line[13:15] == "CB" and line[17:20] != "GLY":
            CB.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])

    crd_CA = torch.tensor(CA, dtype=torch.float)
    crd_CB = torch.tensor(CB, dtype=torch.float)
    dist = torch.cdist(crd_CB, crd_CB)
    return "".join(seq), dist


def compute_lddt(dist_true: torch.Tensor, dist_pred: torch.Tensor, cutoff: float = 15.0):
    mask = (dist_true < cutoff).float() * (1.0 - torch.eye(dist_true.size(0), device=dist_true.device))
    dist_l1 = torch.abs(dist_true - dist_pred)
    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25
    return (score * mask).sum() / mask.sum() 


@click.command()
@click.option(
    "--testset-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="./data/testset/family_0.pkl",
)
@click.option(
    "--output-dir",
    type=click.Path(exists=False, dir_okay=True, path_type=Path),
    required=True,
)
def main(testset_path: Path, output_dir: Path):
    with open(testset_path, "rb") as f:
        testset = pickle.load(f)

    seq_to_cbdist: dict[str, torch.Tensor] = {}
    for data in testset:
        seq_to_cbdist[data["seq"]] = torch.tensor(data["dist"], dtype=torch.float)

    for pdb_path in output_dir.glob("*.pdb"):
        seq, dist = load_pdb(pdb_path)
        lddt = compute_lddt(dist, seq_to_cbdist[seq])     

        print(f"{pdb_path.name}: {lddt:.3f}")


if __name__ == "__main__":
    main()
