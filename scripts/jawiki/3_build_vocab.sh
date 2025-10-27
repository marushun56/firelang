python -m scripts.build_vocab \
    --path_to_corpus='data/corpus/ja-text8/ja-text8.uncased.tokens' \
    --min_count=5 \
    --infreq_replace="<unk>"