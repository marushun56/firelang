wget https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2 -O jawiki-latest-pages-articles.xml.bz2
mkdir -p data/corpus/ja-text8/

# 必要なパッケージをインストール
pip install wikiextractor neologdn

# WikiExtractorで平文テキストを抽出
python -m wikiextractor.WikiExtractor jawiki-latest-pages-articles.xml.bz2 -o ja-text8-extracted --no-templates --processes 8

# 全ての抽出されたテキストを1つのファイルに結合
find ja-text8-extracted -name 'wiki_*' -exec cat {} \; | sed 's/<[^>]*>//g' > data/corpus/ja-text8/ja-text8.raw

# テキストをクリーニング
python scripts/jawiki/clean_wiki.py -i data/corpus/ja-text8/ja-text8.raw -o data/corpus/ja-text8/ja-text8

# 一時ファイルを削除
rm -rf ja-text8-extracted
rm jawiki-latest-pages-articles.xml.bz2
rm data/corpus/ja-text8/ja-text8.raw