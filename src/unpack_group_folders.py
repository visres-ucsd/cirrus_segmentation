from pathlib import Path
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Cirrus Segmentation post-processing')
    parser.add_argument('dir')
    args = parser.parse_args()

    for fp in tqdm(list(Path(args.dir).glob('*/*/*'))):
        parts = list(fp.parts)
        parts = [i for i in parts if not (i=='zhi_controls' or i=='zhi_cases')]
        renamed_path = Path().joinpath(*parts)
        renamed_path.mkdir(parents=True, exist_ok=True)
        if not renamed_path.exists():
            fp.rename(renamed_path)

if __name__=='__main__':
    main()