import pickle
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)

args = parser.parse_args()

path = Path(args.path)

with open(path, "rb") as f:
    results = pickle.load(f)

print(results)
