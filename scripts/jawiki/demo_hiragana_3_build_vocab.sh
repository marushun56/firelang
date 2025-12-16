python -m scripts.build_vocab \
    --path_to_corpus='data/corpus/ja/ja-text8.uncased.hiragana.tokens' \
    --min_count=5 \
    --infreq_replace="<unk>"
