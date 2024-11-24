import pickle
import tarfile
from pathlib import Path
from tqdm.auto import tqdm

import click


def get_pickle_path_by_scop_id(scop_id: str) -> str:
    return f"pkl/{scop_id[1:3]}/{scop_id}.pkl"


@click.command()
@click.option("--split-by", default="family")
@click.option("--split-index", default=0)
@click.option(
    "--split-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="./data/splits.tar.gz",
)
@click.option(
    "--pkl-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="./data/pkl.tar.gz",
)
@click.option(
    "--output-root",
    type=click.Path(exists=True, dir_okay=True, path_type=Path),
    default="./data",
)
@click.option("--limit", default=100, type=int)
def main(split_by: str, split_index: int, split_path: Path, pkl_path: Path, output_root: Path, limit: int):
    # Load splits/{split_by}/{split_index}/valid.txt in splits.tar.gz
    with tarfile.open(split_path, "r:gz") as f:
        scop_ids = f.extractfile(f"splits/{split_by}/{split_index}/valid.txt").read().decode().splitlines()
    print("Number of all data points:", len(scop_ids))
    scop_ids = scop_ids[:limit]
    print("Number of data points to be exported:", len(scop_ids))

    output_path = output_root / f"testset/{split_by}_{split_index}.pkl"
    if output_path.exists():
        click.confirm(f"{output_path} already exists. Overwrite?", abort=True)

    out = []
    with tarfile.open("./data/pkl.tar.gz", "r:gz") as f:
        for scop_id in tqdm(scop_ids):
            try:
                pkl_path = get_pickle_path_by_scop_id(scop_id)
                pkl = f.extractfile(pkl_path).read()
                out.append(pickle.loads(pkl))
            except KeyError:
                print(f"{scop_id} not found in pkl.tar.gz")
                continue

    with open(output_path, "wb") as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    main()
