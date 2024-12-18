import pickle
from pathlib import Path
from tinyfold.models.openfold.residue_constants import restype_3to1

import pandas as pd
import torch
import click
from tqdm.auto import tqdm


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


def download_scop(scop_id: str, dest_path: Path):
    import requests

    url = f"https://scop.berkeley.edu/astral/pdbstyle/ver=2.08&id={scop_id}&output=pdb"
    # Dont check SSL certificate
    response = requests.get(url, verify=False)
    with open(dest_path, "wb") as f:
        f.write(response.content)


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

    seq_to_data: dict[str, torch.Tensor] = {}
    for data in testset:
        seq_to_data[data["seq"]] = data

    results = []
    for pdb_path in output_dir.glob("*.pdb"):
        seq, dist = load_pdb(pdb_path)
        data = seq_to_data[seq]
        dist_true = torch.tensor(data["dist"], dtype=torch.float)
        scope_id = data["scop_id"]
        lddt = compute_lddt(dist, dist_true)
        if "_" in pdb_path.stem:
            prefix = pdb_path.stem.split("_")[0]
        else:
            prefix = "-"
        results.append({
            "prefix": prefix,
            "pdb_path": pdb_path,
            "scop_id": scope_id,
            "lddt": lddt.item(),
            "seq": seq,
        })
        print(f"{pdb_path.name}: {lddt:.3f}")

    df = pd.DataFrame(results)
    df.sort_values("pdb_path", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(df)
    print("Mean LDDT:")
    print(df[["prefix", "lddt"]].groupby("prefix").mean())
    df.to_csv(output_dir / "results.csv", index=False)

    print("Downloading ground truth structures (for visualization only)")
    for scop_id in tqdm(list(df["scop_id"].unique())):
        download_scop(scop_id, output_dir / f"{scop_id}.pdb")


if __name__ == "__main__":
    main()
