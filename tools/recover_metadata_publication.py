"""Inspect a metadata publication; --resume explicitly completes a ready operation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from femto.asset_storage import inspect_publication, recover_publication


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    action = recover_publication if args.resume else inspect_publication
    result = action(args.journal)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
