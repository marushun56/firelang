import argparse
import re
from pathlib import Path
import unicodedata

from tqdm import tqdm
import neologdn
#参考コード
#https://zenn.dev/schnell/scraps/be2c5660b79bdf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input wiki text file")
    parser.add_argument("-o", "--output", required=True, help="Output cleaned file")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    input = Path(args.input)
    output = Path(args.output)
    
    # 基本的な記号のパターン
    basic = re.compile(r"[\u00a1-\u301f]")
    # ハングル文字のパターン
    hangle = re.compile(r"[\uac00-\ud7ff]")
    
    # 入力ファイルの行数を取得（プログレスバー用）
    print(f"Processing {input}...")
    total_lines = sum(1 for _ in input.open())
    
    with input.open() as fin, output.open("w") as fout:
        for line in tqdm(fin, total=total_lines, desc="Cleaning"):
            # neologdnで正規化（半角・全角の統一など）
            line = neologdn.normalize(line)
            # Unicode正規化
            line = unicodedata.normalize("NFKC", line)
            # 不要そうな記号を除去
            line = basic.sub("", line)
            # ハングルを削除
            line = hangle.sub("", line)
            # 短い行は除去
            if len(line) < 50:
                continue
            fout.write(line)
    
    print(f"Cleaned text saved to {output}")


# python scripts/jawiki/clean_wiki.py -i data/corpus/ja-text8/ja-text8.raw -o data/corpus/ja-text8/ja-text8
if __name__ == "__main__":
    main()
