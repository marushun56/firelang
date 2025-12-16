import argparse
import logging
import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.callbacks import CallbackAny2Vec

# Add project root to path
sys.path.append(os.getcwd())

from scripts.benchmark import (
    load_all_word_benchmarks,
    load_all_word_benchmarks_ja,
    ALL_WORDSIM_BENCHMARKS,
    ALL_WORDSIM_BENCHMARKS_JA
)

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class EpochLogger(CallbackAny2Vec):
    def __init__(self, args, benchmarks, tokenizer_func):
        self.epoch = 0
        self.args = args
        self.benchmarks = benchmarks
        self.tokenizer = tokenizer_func

    def on_epoch_end(self, model):
        self.epoch += 1
        logger.info(f"Epoch {self.epoch} finished.")
        if self.epoch % self.args.eval_interval == 0:
            evaluate(model, self.benchmarks, self.tokenizer, self.args)

def get_word_vector(model, word, tokenizer):
    # Tokenize the word using the same logic as training/benchmark loading
    if tokenizer:
        tokens = tokenizer(word)
    else:
        tokens = [word]
    
    valid_vectors = []
    for token in tokens:
        if token in model.wv:
            valid_vectors.append(model.wv[token])
    
    if not valid_vectors:
        return None
    
    # Return the mean vector
    return np.mean(valid_vectors, axis=0)

def evaluate(model, benchmarks, tokenizer, args):
    scores = {}
    logger.info("Starting evaluation...")
    
    for bname, benchmark in benchmarks.items():
        preds = []
        labels = []
        
        # benchmark.texts1 and texts2 contain the raw words
        for t1, t2, sim in zip(benchmark.texts1, benchmark.texts2, benchmark.sims):
            v1 = get_word_vector(model, t1, tokenizer)
            v2 = get_word_vector(model, t2, tokenizer)
            
            if v1 is not None and v2 is not None:
                # Calculate cosine similarity
                cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                preds.append(cos_sim)
                labels.append(sim)
        
        if len(preds) > 0:
            rho, _ = spearmanr(labels, preds)
            scores[bname] = rho * 100
        else:
            scores[bname] = 0.0
            
    # Display results
    df = pd.Series(scores)
    logger.info(f"\n{df.to_string(float_format='%5.1f')}")
    
    # Calculate means
    if args.lang == "both":
        en_scores = df[[b for b in df.index if b in ALL_WORDSIM_BENCHMARKS]]
        ja_scores = df[[b for b in df.index if b in ALL_WORDSIM_BENCHMARKS_JA]]
        mean_en = en_scores.mean() if len(en_scores) > 0 else 0
        mean_ja = ja_scores.mean() if len(ja_scores) > 0 else 0
        logger.info(f"Mean Score: All={df.mean():.2f}, EN={mean_en:.2f}, JA={mean_ja:.2f}")
    else:
        logger.info(f"Mean Score: {df.mean():.2f}")

def main():
    parser = argparse.ArgumentParser(description="Train Word2Vec model and evaluate")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to the tokenized corpus file")
    parser.add_argument("--savedir", type=str, default="./results/word2vec")
    parser.add_argument("--lang", type=str, default="ja", choices=["en", "ja", "both"])
    parser.add_argument("--dim", type=int, default=100, help="Vector dimension")
    parser.add_argument("--win_size", type=int, default=5, help="Window size")
    parser.add_argument("--min_count", type=int, default=5, help="Min count")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--sg", type=int, default=1, help="1 for skip-gram; 0 for CBOW")
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--ja_hiragana", action="store_true", help="Convert Japanese tokens to Hiragana")
    parser.add_argument("--benchmark_lower", type=str, default="True")
    
    args = parser.parse_args()
    args.benchmark_lower = (args.benchmark_lower == "True")
    
    os.makedirs(args.savedir, exist_ok=True)

    # Setup Tokenizer (Copied/Adapted from train.py)
    ja_tokenizer = None
    create_ja_tokenizer = None
    
    if args.lang in ["ja", "both"]:
        ALLOWED_POS_JA = {'動詞', '名詞', '形容詞', '形容動詞', '副詞', '助動詞'}
        def kata2hira(text):
            return "".join([chr(ord(ch) - 96) if ("\u30a1" <= ch <= "\u30f6") else ch for ch in text])

        try:
            import MeCab
            try:
                tagger = MeCab.Tagger()
            except RuntimeError:
                import unidic_lite
                dic_dir = unidic_lite.DICDIR
                tagger = MeCab.Tagger(f"-d {dic_dir}")

            def _create_ja_tokenizer(vocab_set=None):
                def tokenizer(text):
                    tokens = []
                    parsed = tagger.parse(text)
                    for line in parsed.strip().split('\n'):
                        if line == 'EOS' or not line: continue
                        parts = line.split('\t')
                        if len(parts) >= 5:
                            surface = parts[0]
                            reading = parts[1]
                            pos = parts[4].split('-')[0]
                            if pos in ALLOWED_POS_JA:
                                if args.ja_hiragana:
                                    token = kata2hira(reading)
                                    # OOV fallback logic if vocab_set is provided
                                    if vocab_set is not None and token not in vocab_set:
                                        tokens.extend(list(token))
                                    else:
                                        tokens.append(token)
                                else:
                                    tokens.append(surface)
                    return tokens
                return tokenizer
            
            create_ja_tokenizer = _create_ja_tokenizer
            # Initial tokenizer without vocab check
            ja_tokenizer = create_ja_tokenizer(None)
            
        except ImportError:
            logger.warning("MeCab not available")
            def ja_tokenizer(text): return list(text)
            create_ja_tokenizer = lambda x: ja_tokenizer

    # Load Benchmarks
    benchmarks = {}
    eval_tokenizer = None
    
    if args.lang == "en":
        benchmarks = load_all_word_benchmarks(lower=args.benchmark_lower)
        eval_tokenizer = lambda x: [x]
    elif args.lang == "ja":
        benchmarks = load_all_word_benchmarks_ja(lower=False, tokenizer=ja_tokenizer)
        eval_tokenizer = ja_tokenizer
    elif args.lang == "both":
        b_en = load_all_word_benchmarks(lower=args.benchmark_lower)
        b_ja = load_all_word_benchmarks_ja(lower=False, tokenizer=ja_tokenizer)
        benchmarks = {**b_en, **b_ja}
        eval_tokenizer = ja_tokenizer 
    
    logger.info(f"Loading corpus from {args.corpus_path}")
    sentences = LineSentence(args.corpus_path)
    
    logger.info("Building vocab...")
    model = Word2Vec(
        vector_size=args.dim,
        window=args.win_size,
        min_count=args.min_count,
        workers=args.workers,
        sg=args.sg
    )
    model.build_vocab(sentences)
    
    # Update tokenizer with vocab for OOV fallback
    if args.lang in ["ja", "both"] and create_ja_tokenizer:
        vocab_set = set(model.wv.index_to_key)
        eval_tokenizer = create_ja_tokenizer(vocab_set)

    logger.info("Training...")
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=args.epochs,
        callbacks=[EpochLogger(args, benchmarks, eval_tokenizer)]
    )
    
    save_path = os.path.join(args.savedir, "word2vec.model")
    model.save(save_path)
    logger.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
