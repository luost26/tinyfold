import click
import torch
import pathlib
from matplotlib import pyplot as plt

from tinyfold.models.esmfold import ESMFold


@click.command()
@click.option("--device", default="cuda")
@click.option("esm_path", "--esm", type=click.Path(exists=True), default="./data/esm2_t36_3B_UR50D.pt")
@click.option(
    "esmfold_path",
    "--esmfold",
    type=click.Path(exists=True),
    default="./data/esmfold_structure_module_only_3B.pt",
)
@click.option("output_dir", "--out", type=click.Path(path_type=pathlib.Path), required=True)
def main(esm_path: str, esmfold_path: str, device: str, output_dir: pathlib.Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {esmfold_path} and {esm_path}")
    model = ESMFold.load(esmfold_path, esm_path).to(device)
    model.eval()

    torch.set_grad_enabled(False)

    cnt = 0
    while True:
        cnt += 1
        out_pdb_path = output_dir / f"{cnt:04d}.pdb"
        if out_pdb_path.exists():
            continue

        seq = click.prompt("SEQ").strip()
        out = model.infer(seq)

        plddt = out["mean_plddt"].item()
        print(f"pLDDT: {plddt:.2f}")

        p_ca = out["positions"][-1, 0, :, 1]
        dist = torch.cdist(p_ca, p_ca).cpu().numpy()

        plt.imshow(dist)
        plt.savefig(output_dir / f"{cnt:04d}.png")
        with open(out_pdb_path, "w") as f:
            f.writelines(model.output_to_pdb(out))
        print(f"Saved to {out_pdb_path}")


if __name__ == "__main__":
    main()
