import os

def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--raw_path", type=str, default="data/corpus/text8/text8")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--cased", action="store_true")
    parser.add_argument("--sos", type=str, default="<s>")
    parser.add_argument("--eos", type=str, default="</s>")
    parser.add_argument("--linesep", type=str, default="\n")
    parser.add_argument("--tokensep", type=str, default=" ")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "ja"], 
                        help="Language for tokenization (en: English with NLTK, ja: Japanese with MeCab/Janome)")
    parser.add_argument("--ja_tokenizer", type=str, default="mecab", choices=["mecab", "janome"],
                        help="Japanese tokenizer to use (mecab or janome)")
    parser.add_argument("--ja_pos_filter", action="store_true",
                        help="Filter Japanese tokens by POS (noun, verb, adjective, adverb, auxiliary verb)")

    return parser.parse_args()


def run(args):
    from nltk import word_tokenize
    import tqdm
    from scripts.multiproc_yield import yielder
    import logging
    import json

    logger = logging.getLogger()
    path = os.path.abspath(args.raw_path)
    dirpath = os.path.dirname(path)
    filename = os.path.basename(path)
    savepath = (
        f"{dirpath}/{filename}" + ("" if args.cased else ".uncased") + ".tokens"
    )
    logger.info(f"Tokenized corpus to be saved at {savepath}")

    argsavepath = savepath + '.args'
    with open(argsavepath, "wt") as f:
        json.dump(args.__dict__, f, indent=2)
    logger.info(f"Arguments saved at {argsavepath}")

    def run_tokenize(corpus_path, save_path, args, max_size_bytes=1024 * 1024 * 1024):
        logger.info("Tokenizing...")

        # 日本語トークナイザーの初期化
        ja_tokenizer_instance = None
        ja_pos_filter = getattr(args, 'ja_pos_filter', False)
        
        # 許可する品詞（動詞、名詞、形容詞、形容動詞、副詞、助動詞）
        ALLOWED_POS = {'動詞', '名詞', '形容詞', '形容動詞', '副詞', '助動詞'}
        
        if args.lang == "ja":
            if args.ja_tokenizer == "mecab":
                try:
                    import MeCab
                    # 辞書パスを自動検出して指定
                    try:
                        # 品詞フィルタを使う場合は -Ochasen か デフォルト出力を使う
                        if ja_pos_filter:
                            ja_tokenizer_instance = MeCab.Tagger()
                        else:
                            ja_tokenizer_instance = MeCab.Tagger("-Owakati")
                    except RuntimeError:
                        # unidic-liteを使用
                        import unidic_lite
                        dic_dir = unidic_lite.DICDIR
                        if ja_pos_filter:
                            ja_tokenizer_instance = MeCab.Tagger(f"-d {dic_dir}")
                        else:
                            ja_tokenizer_instance = MeCab.Tagger(f"-Owakati -d {dic_dir}")
                    logger.info(f"Using MeCab for Japanese tokenization (POS filter: {ja_pos_filter})")
                except ImportError as e:
                    logger.error(f"MeCab not installed: {e}")
                    raise
            elif args.ja_tokenizer == "janome":
                try:
                    from janome.tokenizer import Tokenizer
                    ja_tokenizer_instance = Tokenizer()
                    logger.info(f"Using Janome for Japanese tokenization (POS filter: {ja_pos_filter})")
                except ImportError:
                    logger.error("Janome not installed. Please install janome")
                    raise

        def _tokenize(line, sos, eos, lang, ja_tok=None, pos_filter=False, allowed_pos=None):
            if not args.cased:
                line = line.lower()
            line = line.strip()
            if not line:
                return []
            
            if lang == "ja":
                # 日本語のトークナイズ
                if args.ja_tokenizer == "mecab":
                    if pos_filter and allowed_pos:
                        # 品詞フィルタ付きトークナイズ
                        tokens = []
                        parsed = ja_tok.parse(line)
                        for line_result in parsed.strip().split('\n'):
                            if line_result == 'EOS' or not line_result:
                                continue
                            parts = line_result.split('\t')
                            # UniDic形式: 表層形, 読み1, 読み2, 原形, 品詞, ...
                            # 品詞は5番目（index 4）にある（例: "名詞-固有名詞-地名-一般"）
                            if len(parts) >= 5:
                                surface = parts[0]
                                pos_full = parts[4]  # 品詞情報（例: "動詞-非自立可能"）
                                pos = pos_full.split('-')[0]  # 最初の品詞だけ取得
                                if pos in allowed_pos:
                                    tokens.append(surface)
                    else:
                        tokens = ja_tok.parse(line).strip().split()
                elif args.ja_tokenizer == "janome":
                    if pos_filter and allowed_pos:
                        tokens = [token.surface for token in ja_tok.tokenize(line)
                                  if token.part_of_speech.split(',')[0] in allowed_pos]
                    else:
                        tokens = [token.surface for token in ja_tok.tokenize(line)]
                return [sos] + tokens + [eos]
            else:
                # 英語のトークナイズ
                return [sos] + word_tokenize(line) + [eos]

        def _linecutter(lines, maxlen=10000):
            for line in lines:
                nseg = (len(line) + maxlen - 1) // maxlen
                for i in range(nseg):
                    yield line[i * maxlen : (i + 1) * maxlen]

        ntokens = 0
        fout = open(save_path, "wt")
        with open(corpus_path, "rt") as f:
            for tokens in tqdm.tqdm(
                yielder(
                    _linecutter(f),
                    _tokenize,
                    num_workers=args.num_workers,
                    additional_kwds={
                        "sos": args.sos, 
                        "eos": args.eos,
                        "lang": args.lang,
                        "ja_tok": ja_tokenizer_instance,
                        "pos_filter": ja_pos_filter,
                        "allowed_pos": ALLOWED_POS
                    },
                    max_size_bytes=max_size_bytes,
                )
            ):
                if tokens:
                    ntokens += len(tokens)
                    fout.write(args.tokensep.join(tokens) + args.linesep)
        logger.info(f"{ntokens} tokens in saved.")

        fout.close()

    run_tokenize(path, savepath, args)
    logger.info("Finished.")


if __name__ == "__main__":
    args = parse_args()
    run(args)
