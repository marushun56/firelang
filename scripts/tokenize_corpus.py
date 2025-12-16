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
    parser.add_argument("--ja_hiragana", action="store_true",
                        help="Convert Japanese tokens to Hiragana (using reading)")

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
        f"{dirpath}/{filename}" + ("" if args.cased else ".uncased") + 
        (".hiragana" if getattr(args, 'ja_hiragana', False) else "") + ".tokens"
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
        ja_hiragana = getattr(args, 'ja_hiragana', False)

        def kata2hira(text):
            return "".join([chr(ord(ch) - 96) if ("\u30a1" <= ch <= "\u30f6") else ch for ch in text])
        
        # 許可する品詞（動詞、名詞、形容詞、形容動詞、副詞、助動詞）
        ALLOWED_POS = {'動詞', '名詞', '形容詞', '形容動詞', '副詞', '助動詞'}
        
        if args.lang == "ja":
            if args.ja_tokenizer == "mecab":
                try:
                    import MeCab
                    # 辞書パスを自動検出して指定
                    try:
                        # 品詞フィルタを使う場合は -Ochasen か デフォルト出力を使う
                        if ja_pos_filter or ja_hiragana:
                            ja_tokenizer_instance = MeCab.Tagger()
                        else:
                            ja_tokenizer_instance = MeCab.Tagger("-Owakati")
                    except RuntimeError:
                        # unidic-liteを使用
                        import unidic_lite
                        dic_dir = unidic_lite.DICDIR
                        if ja_pos_filter or ja_hiragana:
                            ja_tokenizer_instance = MeCab.Tagger(f"-d {dic_dir}")
                        else:
                            ja_tokenizer_instance = MeCab.Tagger(f"-Owakati -d {dic_dir}")
                    logger.info(f"Using MeCab for Japanese tokenization (POS filter: {ja_pos_filter}, Hiragana: {ja_hiragana})")
                except ImportError as e:
                    logger.error(f"MeCab not installed: {e}")
                    raise
            elif args.ja_tokenizer == "janome":
                try:
                    from janome.tokenizer import Tokenizer
                    ja_tokenizer_instance = Tokenizer()
                    logger.info(f"Using Janome for Japanese tokenization (POS filter: {ja_pos_filter}, Hiragana: {ja_hiragana})")
                except ImportError:
                    logger.error("Janome not installed. Please install janome")
                    raise

        def _tokenize(line, sos, eos, lang, ja_tok=None, pos_filter=False, allowed_pos=None, to_hiragana=False):
            if not args.cased:
                line = line.lower()
            line = line.strip()
            if not line:
                return []
            
            if lang == "ja":
                # 日本語のトークナイズ
                if args.ja_tokenizer == "mecab":
                    if pos_filter or to_hiragana:
                        # 品詞フィルタまたはひらがな変換付きトークナイズ
                        tokens = []
                        parsed = ja_tok.parse(line)
                        for line_result in parsed.strip().split('\n'):
                            if line_result == 'EOS' or not line_result:
                                continue
                            parts = line_result.split('\t')
                            
                            # 既存のロジックに従い、フォーマットチェックを行う
                            # UniDic形式: 表層形, 読み1, 読み2, 原形, 品詞, ...
                            if len(parts) < 5:
                                # フォーマットが期待通りでない場合
                                # pos_filterが有効ならスキップ（既存動作）
                                if pos_filter:
                                    continue
                                # to_hiraganaのみの場合、読みが取れないので表層形
                                tokens.append(parts[0])
                                continue

                            surface = parts[0]
                            reading = parts[1]
                            pos_full = parts[4]
                            
                            if pos_filter and allowed_pos:
                                pos = pos_full.split('-')[0]
                                if pos not in allowed_pos:
                                    continue
                            
                            if to_hiragana:
                                tokens.append(kata2hira(reading))
                            else:
                                tokens.append(surface)
                    else:
                        tokens = ja_tok.parse(line).strip().split()
                elif args.ja_tokenizer == "janome":
                    tokens = []
                    for token in ja_tok.tokenize(line):
                        if pos_filter and allowed_pos:
                            if token.part_of_speech.split(',')[0] not in allowed_pos:
                                continue
                        
                        if to_hiragana:
                            r = token.reading
                            if r and r != '*':
                                tokens.append(kata2hira(r))
                            else:
                                tokens.append(token.surface)
                        else:
                            tokens.append(token.surface)
                    
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
                        "allowed_pos": ALLOWED_POS,
                        "to_hiragana": ja_hiragana
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
