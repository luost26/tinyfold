import random
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
@click.option("--limit", default=10, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--min-length", default=100, type=int)
@click.option("--max-length", default=400, type=int)
def main(
    split_by: str,
    split_index: int,
    split_path: Path,
    pkl_path: Path,
    output_root: Path,
    limit: int,
    seed: int,
    min_length: int,
    max_length: int,
):
    # Load splits/{split_by}/{split_index}/valid.txt in splits.tar.gz
    with tarfile.open(split_path, "r:gz") as f:
        scop_ids = f.extractfile(f"splits/{split_by}/{split_index}/valid.txt").read().decode().splitlines()
    print("Number of all data points:", len(scop_ids))
    print("Max number of data points to export:", limit)
    print(f"Length limit: [{min_length}, {max_length}]")

    scop_ids.sort()
    random.Random(seed).shuffle(scop_ids)

    output_path = output_root / f"testset/{split_by}_{split_index}.pkl"
    outseq_path = output_root / f"testset/{split_by}_{split_index}_sequences.txt"
    if output_path.exists():
        click.confirm(f"{output_path} already exists. Overwrite?", abort=True)
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    out = []
    with tarfile.open("./data/pkl.tar.gz", "r:gz") as f:
        pbar = tqdm(total=limit)
        with pbar:
            for scop_id in scop_ids:
                try:
                    pkl_path = get_pickle_path_by_scop_id(scop_id)
                    pkl = f.extractfile(pkl_path).read()
                    data = pickle.loads(pkl)
                    seqlen = len(data["seq"])
                    if seqlen < min_length or seqlen > max_length:
                        print(f"{scop_id} is too short/long ({seqlen}), ignoring")
                        continue
                    data["scop_id"] = scop_id
                    out.append(data)
                    pbar.update(1)
                    if len(out) >= limit:
                        break
                except KeyError:
                    print(f"{scop_id} not found in pkl.tar.gz, ignoring")
                    continue

    with open(output_path, "wb") as f:
        pickle.dump(out, f)

    with open(outseq_path, "w") as f:
        for data in out:
            f.write(data["seq"] + "\n")

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
