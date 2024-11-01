import click

from efficient_esm.models.esm2 import ESM2


@click.command()
@click.option("--device", default="cuda")
@click.option("model_path", "--model", type=click.Path(exists=True), default="./data/esm2_t36_3B_UR50D.pt")
def main(model_path: str, device: str):
    model = ESM2.load(model_path).to(device)
    print(f"Loaded model from {model_path} on {device}")


if __name__ == "__main__":
    main()
