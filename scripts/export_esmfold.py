import click
import pathlib
import shutil

from tinyfold.models.esmfold import ESMFold

@click.command()
@click.option("esm_path", "--esm", type=click.Path(exists=True), default="./data/esm2_t36_3B_UR50D.pt")
@click.option(
    "esmfold_path",
    "--esmfold",
    type=click.Path(exists=True),
    default="./data/esmfold_structure_module_only_3B.pt",
)
@click.option("output_dir", "--out", type=click.Path(path_type=pathlib.Path), default="./data/esmfold_fp32")
def main(esm_path: str, esmfold_path: str, output_dir: pathlib.Path):
    if output_dir.exists():
        click.confirm(f"{output_dir} already exists. Overwrite?", abort=True)
        shutil.rmtree(output_dir)
    print(f"Loading model from {esmfold_path} and {esm_path}")
    model = ESMFold.load(esmfold_path, esm_path)
    print("Exporting model")
    model.export(output_dir)


if __name__ == "__main__":
    main()
