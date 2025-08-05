import argparse
import json
import os
import subprocess
import sys

def main():
    p = argparse.ArgumentParser(
        description="Resume training on a specific train_subset_index."
    )
    p.add_argument(
        "train_subset_index",
        type=int,
        help="Which subset index to resume (as in splits/run_mapping.json)."
    )
    p.add_argument(
        "--config_path",
        default="model_babylm_bert.json",
        help="Path to your training config JSON."
    )
    args = p.parse_args()

    # load config to find cache_path
    with open(args.config_path, "r") as cf:
        config = json.load(cf)
    cache_path = config["cache_path"]

    # locate run mapping
    runs_base    = os.path.join(cache_path, "runs")
    mapping_file = os.path.join(runs_base, "run_mapping.json")
    if not os.path.exists(mapping_file):
        print(f"ERROR: run mapping not found at {mapping_file}", file=sys.stderr)
        sys.exit(1)

    mapping = json.load(open(mapping_file))
    key = str(args.train_subset_index)
    if key not in mapping:
        print(f"ERROR: no entry for subset index {key} in {mapping_file}", file=sys.stderr)
        sys.exit(1)

    # build full checkpoint path and verify
    run_dir = os.path.join("./", mapping[key])
    if not os.path.isdir(run_dir):
        print(f"ERROR: checkpoint folder not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Resuming subset {key} from checkpoint at:\n  {run_dir}\n")

    # invoke trainer with override checkpoint_path
    cmd = [
        sys.executable, "transformer_trainer.py",
        "--config_path", args.config_path,
        "--checkpoint_path", run_dir
    ]
    ret = subprocess.call(cmd)
    if ret != 0:
        print(f"trainer exited with code {ret}", file=sys.stderr)
    sys.exit(ret)


if __name__ == "__main__":
    main()