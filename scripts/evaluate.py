import argparse
import os
import torch
import numpy as np
import pandas as pd
from firelang import FireWord
from firelang.utils.log import logger
from scripts.benchmark import (
    ALL_WORDSIM_BENCHMARKS,
    ALL_WORDSIM_BENCHMARKS_JA,
    load_all_word_benchmarks,
    load_all_word_benchmarks_ja,
    benchmark_word_similarity,
    benchmark_word_similarity_ja,
)

def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained FireWord model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory or checkpoint.")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "ja", "both"], help="Language for benchmarking.")
    parser.add_argument("--benchmark_lower", type=boolean_string, default="True", help="Convert benchmark texts to lower case.")
    parser.add_argument("--ja_hiragana", action="store_true", help="Convert Japanese tokens to Hiragana (using reading)")
    parser.add_argument("--cpu", action="store_true", help="Use cpu rather than CUDA.")
    
    args = parser.parse_args()
    
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    try:
        model = FireWord.from_pretrained(args.model_path)
    except Exception as e:
        # Try appending 'best' if it's a directory
        best_path = os.path.join(args.model_path, "best")
        if os.path.exists(best_path):
             logger.info(f"Trying to load from {best_path}")
             model = FireWord.from_pretrained(best_path)
        else:
            raise e
            
    model = model.to(device)
    model.eval()

    # Setup benchmarks
    benchmarks = {}
    
    # Japanese Tokenizer Logic
    ALLOWED_POS_JA = {'動詞', '名詞', '形容詞', '形容動詞', '副詞', '助動詞'}
    def kata2hira(text):
        return "".join([chr(ord(ch) - 96) if ("\u30a1" <= ch <= "\u30f6") else ch for ch in text])

    try:
        import MeCab
        # Try to initialize MeCab with unidic-lite if default fails (similar to tokenize_corpus.py)
        try:
            tagger = MeCab.Tagger()
        except RuntimeError:
            import unidic_lite
            dic_dir = unidic_lite.DICDIR
            tagger = MeCab.Tagger(f"-d {dic_dir}")

        def create_smart_tokenizer(vocab):
            # Determine vocab interface
            if hasattr(vocab, 's2i'):
                s2i = vocab.s2i
            elif hasattr(vocab, 'stoi'):
                s2i = vocab.stoi
            else:
                s2i = {} # Fallback
            
            def ja_tokenizer(text):
                tokens = []
                parsed = tagger.parse(text)
                for line in parsed.strip().split('\n'):
                    if line == 'EOS' or not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 5:
                        surface = parts[0]
                        reading = parts[1]
                        pos_full = parts[4]
                        pos = pos_full.split('-')[0]
                        if pos in ALLOWED_POS_JA:
                            if args.ja_hiragana:
                                token_candidate = kata2hira(reading)
                                # Smart OOV handling: if token is OOV, split into characters
                                if token_candidate in s2i:
                                    tokens.append(token_candidate)
                                else:
                                    # Fallback to character level
                                    tokens.extend(list(token_candidate))
                            else:
                                tokens.append(surface)
                return tokens
            return ja_tokenizer
            
    except ImportError:
        logger.warning("MeCab not available, using character-level tokenization")
        def create_smart_tokenizer(vocab):
            def ja_tokenizer(text):
                return list(text)
            return ja_tokenizer

    # Create tokenizer with model vocab
    ja_tokenizer = create_smart_tokenizer(model.vocab)

    if args.lang == "en":
        benchmarks = load_all_word_benchmarks(lower=args.benchmark_lower)
    elif args.lang == "ja":
        benchmarks = load_all_word_benchmarks_ja(
            lower=False,
            tokenizer=ja_tokenizer
        )
    elif args.lang == "both":
        benchmarks_en = load_all_word_benchmarks(lower=args.benchmark_lower)
        benchmarks_ja = load_all_word_benchmarks_ja(
            lower=False,
            tokenizer=ja_tokenizer
        )
        benchmarks = {**benchmarks_en, **benchmarks_ja}

    logger.info(f"Loaded {len(benchmarks)} benchmarks.")

    simscores = pd.Series(dtype=float)

    # English benchmarks (Word Similarity)
    if args.lang in ["en", "both"]:
        benchmarks_en_to_run = {k: v for k, v in benchmarks.items() if k in ALL_WORDSIM_BENCHMARKS}
        if benchmarks_en_to_run:
            scores_en = benchmark_word_similarity(model, benchmarks_en_to_run) * 100
            simscores = pd.concat([simscores, scores_en])

    # Japanese benchmarks (Word Similarity with morpheme-aware calculation)
    if args.lang in ["ja", "both"]:
        benchmarks_ja_to_run = {k: v for k, v in benchmarks.items() if k in ALL_WORDSIM_BENCHMARKS_JA}
        if benchmarks_ja_to_run:
            scores_ja = benchmark_word_similarity_ja(model, benchmarks_ja_to_run) * 100
            simscores = pd.concat([simscores, scores_ja])

    print("\nBenchmark Results:")
    print(simscores.to_string(float_format="%5.1f"))
    print(f"\nMean Score: {simscores.mean():.1f}")

if __name__ == "__main__":
    main()
