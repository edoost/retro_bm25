import argparse
import os 
from pathlib import Path


def main(args):
    files = os.listdir(args.parent)

    for f in files:
        n = f.split('_')[0]
        p = Path(args.parent / Path('split_' + str(n)))
        p.mkdir(exist_ok=True)
        Path(args.parent / f).rename(p / f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, help='Path to val_spec_data', required=True)
    args = parser.parse_args()
    main(args)
