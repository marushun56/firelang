from __future__ import annotations
from typing import List
from string import ascii_uppercase, digits
from contextlib import nullcontext
import argparse
import os
import io
import random
import json
import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW, Adam, SGD, Adagrad
from torch.optim.lr_scheduler import OneCycleLR

from corpusit import SkipGramConfigWithTokenization
from firelang import FireWord, FireWordConfig, FireTensor
from firelang.utils.optim import DummyScheduler
from firelang.utils.log import logger
from firelang.utils.timer import elapsed, Timer
from scripts.benchmark import (
    SimilarityBenchmark,
    SimilarityBenchmark,
    ALL_WORDSIM_BENCHMARKS,
    ALL_WORDSIM_BENCHMARKS_JA,
    load_word_benchmark,
    load_all_word_benchmarks,
    load_all_word_benchmarks_ja,
    benchmark_word_similarity,
    benchmark_word_similarity_ja,
    benchmark_sentence_similarity,
)
from scripts.dataloader import DataLoader

matplotlib.use("Agg")


logger.setLevel(level=os.environ.get("LOGLEVEL", "DEBUG").upper())

try:
    import wandb
    from wandb_config import download_wandb_files
except Exception as e:
    logger.warning(
        "Unable to import wandb for experiment tracking. "
        "Consider executing `pip install wandb`."
    )

total_timer = Timer(elapsed, "total")


@total_timer
def train(args):
    torch.set_num_threads(1)
    if args.use_wandb:
        from wandb_config import config

        config()
        exp_name = "".join(random.choices(ascii_uppercase + digits, k=8))
        args.savedir = f"{args.savedir}/{exp_name}/"
        wandb.init(name=exp_name, config=args.__dict__)

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"

    if args.task == "skipgram":
        
        # Try to load vocab and word counts
        vocab_path = args.corpus_path + ".vocab.json"
        word_counts_path = args.corpus_path + ".word_counts.json"
        
        if not os.path.exists(vocab_path):
            raise ValueError(
                f"Vocab file not found at {vocab_path}. "
                "You need to build the vocabulary first using build_vocab.py."
            )
        if not os.path.exists(word_counts_path):
            raise ValueError(
                f"Word counts file not found at {word_counts_path}. "
                "You need to build the vocabulary first using build_vocab.py."
            )
        
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        
        # Load word counts
        with open(word_counts_path, 'r') as f:
            word_counts_data = json.load(f)
        
        # Convert string keys to integers for word_counts
        word_counts = {int(k): v for k, v in word_counts_data.items()}
        
        # Create word_to_id mapping
        word_to_id = vocab_data['s2i']
        
        # Create dataset configuration
        config = SkipGramConfigWithTokenization(
            word_counts=word_counts,
            word_to_id=word_to_id,
            separator=" ",
            win_size=args.win_size,
            subsample=args.subsample,
            power=args.power,
            n_neg=args.n_neg,
        )
        
        # Create sampler
        dataset = config.sampler(args.seed, num_threads=4)
        
        # Convert i2s keys to integers
        i2s = {int(k): v for k, v in vocab_data['i2s'].items()}
        
        vocab = SimpleVocab(vocab_data['s2i'], i2s, word_counts=word_counts)
    else:
        raise ValueError(f"Failed to recognize task == {args.task}")
    # Create a more efficient dataloader
    
    dataloader = DataLoader(dataset, args.sz_batch, args.corpus_path, args.read_mode)
    logger.info(f"Dataset initialized with a dictionary of size {len(vocab)} .")

    set_seed(args.seed)
    if not (args.use_wandb and args.wandb_pretrained):
        config = FireWordConfig(dim=args.dim, func=args.func, measure=args.measure)
        if args.model.lower() == "fireword":
            model = FireWord(config=config, vocab=vocab)
        else:
            raise ValueError(args.model)
    else:
        if args.model.lower() == "fireword":
            model = FireWord.from_pretrained(
                download_wandb_files(args.wandb_pretrained, "best")
            )
        wandb.config.update({"continue": True, "previous_run": args.wandb_pretrained})
    logger.info(model)
    num_parameters = count_parameters(model) // len(vocab)
    logger.info(f"number of parameters for each word: {num_parameters}")
    if args.use_wandb:
        wandb.log({"num_parameters": num_parameters, "vocab_size": len(vocab)})

    model = model.to(device)
    if args.optimizer == "adamw":
        optimizer = AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adagrad":
        optimizer = Adagrad(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_scheduler == "OneCycleLR":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=args.n_iters // args.accum_steps + 5,
            div_factor=1.0,
            final_div_factor=20.0,
        )
    else:
        scheduler = DummyScheduler(args.lr)
    logger.info(f"Initialized optimizer and scheduler")
    logger.info(f"  Optimizer: {optimizer}")
    logger.info(f"  Scheduler: {scheduler}")

    # ベンチマークの選択
    viz_tokenizer = None
    if args.lang == "en":
        benchmark_list = ALL_WORDSIM_BENCHMARKS
        load_benchmarks_func = load_all_word_benchmarks
        benchmark_kwargs = {"lower": args.benchmark_lower}
    elif args.lang == "ja":
        benchmark_list = ALL_WORDSIM_BENCHMARKS_JA
        load_benchmarks_func = load_all_word_benchmarks_ja
        # 日本語の場合はコーパスと同じトークナイザーを使う（品詞フィルタ付き）
        # 許可する品詞（動詞、名詞、形容詞、形容動詞、副詞、助動詞）
        ALLOWED_POS_JA = {'動詞', '名詞', '形容詞', '形容動詞', '副詞', '助動詞'}
        try:
            import MeCab
            tagger = MeCab.Tagger()
            def ja_tokenizer(text):
                tokens = []
                parsed = tagger.parse(text)
                for line in parsed.strip().split('\n'):
                    if line == 'EOS' or not line:
                        continue
                    parts = line.split('\t')
                    # UniDic形式: 表層形, 読み1, 読み2, 原形, 品詞, ...
                    # 品詞は5番目（index 4）にある
                    if len(parts) >= 5:
                        surface = parts[0]
                        pos_full = parts[4]  # 品詞情報
                        pos = pos_full.split('-')[0]  # 最初の品詞だけ取得
                        if pos in ALLOWED_POS_JA:
                            tokens.append(surface)
                return tokens
        except ImportError:
            logger.warning("MeCab not available, using character-level tokenization")
            def ja_tokenizer(text):
                return list(text)
        viz_tokenizer = ja_tokenizer
        benchmark_kwargs = {"lower": args.benchmark_lower, "tokenizer": ja_tokenizer}
    elif args.lang == "both":
        benchmark_list = ALL_WORDSIM_BENCHMARKS + ALL_WORDSIM_BENCHMARKS_JA
        # 両方のベンチマークをロード
        benchmarks_en = load_all_word_benchmarks(lower=args.benchmark_lower)
        
        # 日本語用に適切なトークナイザーを使用（英語と同じ）
        # 許可する品詞（動詞、名詞、形容詞、形容動詞、副詞、助動詞）
        ALLOWED_POS_JA = {'動詞', '名詞', '形容詞', '形容動詞', '副詞', '助動詞'}
        try:
            import MeCab
            tagger = MeCab.Tagger()
            def ja_tokenizer(text):
                tokens = []
                parsed = tagger.parse(text)
                for line in parsed.strip().split('\n'):
                    if line == 'EOS' or not line:
                        continue
                    parts = line.split('\t')
                    # UniDic形式: 表層形, 読み1, 読み2, 原形, 品詞, ...
                    # 品詞は5番目（index 4）にある
                    if len(parts) >= 5:
                        surface = parts[0]
                        pos_full = parts[4]  # 品詞情報
                        pos = pos_full.split('-')[0]  # 最初の品詞だけ取得
                        if pos in ALLOWED_POS_JA:
                            tokens.append(surface)
                return tokens
        except ImportError:
            logger.warning("MeCab not available for Japanese benchmark, using character-level")
            def ja_tokenizer(text):
                return list(text)
        viz_tokenizer = ja_tokenizer
        
        benchmarks_ja = load_all_word_benchmarks_ja(
            lower=False,  # 日本語には大文字・小文字の概念がないのでFalse
            tokenizer=ja_tokenizer
        )
        benchmarks = {**benchmarks_en, **benchmarks_ja}
    else:
        raise ValueError(f"Invalid lang: {args.lang}")
    
    # lang != "both"の場合のベンチマークロード
    if args.lang != "both":
        benchmarks = load_benchmarks_func(**benchmark_kwargs)

    logger.info(f"Loaded {len(benchmarks)} benchmarks for language: {args.lang}")

    best_iter, best_simscore, best_loss = -1, 0, float("Inf")
    best_savepath = f"{args.savedir}/best"

    if args.profile:
        from torch.profiler import profile, ProfilerActivity
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
    else:
        prof = nullcontext()

    for i in range(1, args.n_iters + 1):

        with Timer(elapsed, "prepare", sync_cuda=True):
            inputs, labels = next(dataloader)
            inputs = torch.tensor(
                inputs, dtype=torch.long, device=device
            )  # (sz_batch, 2) int
            # Convert numpy bool to Python bool to avoid compatibility issues
            labels = [bool(label) for label in labels]
            labels = torch.tensor(
                labels, dtype=torch.bool, device=device
            )  # labels: (sz_batch, ) bool
        """ ----------------- forward pass -------------------"""

        with prof, Timer(elapsed, "forward", sync_cuda=True):
            n_pos = labels.sum().item()
            n_neg = (~labels).sum().item()

            if args.task == "skipgram":
                if args.model.lower() == "fireword":
                    model: FireWord
                    loss = model.loss_skipgram(
                        inputs,
                        labels,
                        args,
                    )
                else:
                    raise ValueError(args.model)
            else:
                raise ValueError(f"Failed to recognize task == {args.task}")
            total_loss = loss.reduced_total()
            steploss = total_loss / args.accum_steps
        if args.profile:
            logger.debug("----- forward -----")
            logger.debug(prof.key_averages().table(sort_by="self_cpu_time_total"))
        """ ----------------- backward pass -------------------"""
        with prof, Timer(elapsed, "backward", sync_cuda=True):
            steploss.backward()

            grad_norm = (
                torch.cat([p.grad.data.reshape(-1) for p in model.parameters()])
                .norm()
                .item()
            )

        if args.profile:
            logger.debug("----- backward -----")
            logger.debug(prof.key_averages().table(sort_by="self_cpu_time_total"))
        """ ----------------- optim -------------------"""
        if i % args.accum_steps == 0:
            with Timer(elapsed, "optim", sync_cuda=True):
                with Timer(elapsed, "step"):
                    for name, p in model.named_parameters():
                        isnan = p.grad.isnan()
                        isinf = p.grad.isinf()
                        isinvalid = isnan | isinf
                        if isinvalid.any():
                            p.grad.masked_fill_(isinvalid, 0)
                            p[isinvalid].normal_(0, 0.1)
                            print(f"Fixed nan/inf values in grad of {name}")
                            print(f"  grad = {p.grad}")

                    optimizer.step()

                with Timer(elapsed, "lrstep", sync_cuda=True):
                    scheduler.step()

                with Timer(elapsed, "zerograd", sync_cuda=True):
                    model.zero_grad()

        if i % args.eval_interval == 0:

            os.makedirs(args.savedir, exist_ok=True)
            model.eval()

            """--------------- similarity benchmark ---------------"""
            with Timer(elapsed, "benchmark", sync_cuda=True):
                if args.model.lower() == "fireword":
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
                else:
                    raise ValueError(args.model)
            
            # 言語別の平均スコアを計算
            if args.lang == "en":
                simscore = simscores.mean()
                simscore_en = simscore
                simscore_ja = None
            elif args.lang == "ja":
                simscore = simscores.mean()
                simscore_en = None
                simscore_ja = simscore
            elif args.lang == "both":
                # ENとJAのスコアを分けて計算
                en_scores = simscores[[b for b in simscores.index if b in ALL_WORDSIM_BENCHMARKS]]
                ja_scores = simscores[[b for b in simscores.index if b in ALL_WORDSIM_BENCHMARKS_JA]]
                simscore_en = en_scores.mean() if len(en_scores) > 0 else None
                simscore_ja = ja_scores.mean() if len(ja_scores) > 0 else None
                simscore = simscores.mean()
            
            if simscore > best_simscore:
                best_iter = i
                best_simscore = simscore
                best_loss = total_loss.item()
                model.save(best_savepath)

            # ログ出力を言語別に対応
            log_msg = (
                f"Iter {i}. Loss={loss}; grad={grad_norm:.3g}; "
                f"lr={scheduler.get_last_lr()[0]:.3g}; "
                f"n={n_pos}+{n_neg}; "
            )
            if args.lang == "both":
                log_msg += f"meansim(all)={simscore:.3f}%; "
                if simscore_en is not None:
                    log_msg += f"EN={simscore_en:.3f}%; "
                if simscore_ja is not None:
                    log_msg += f"JA={simscore_ja:.3f}%"
            else:
                log_msg += f"meansim={simscore:.3f}%"
            
            logger.info(log_msg)
            logger.debug(simscores.to_string(float_format="%5.1f"))
            total_timer.update()
            logger.debug("-- Elapsed --\n" + elapsed.format(thresh=0.8))

            if args.use_wandb:
                loginfo = {
                    "iter": i,
                    "eval/loss": total_loss.item(),
                    **{f"eval/loss/{ln}": l.item() for ln, l in loss.reduced_items()},
                    "eval/grad": grad_norm,
                    "eval/simscore": simscore,
                    **{f"eval/simscore/{bn}": score for bn, score in simscores.items()},
                }
                
                # 言語別スコアをwandbに記録
                if args.lang == "both":
                    if simscore_en is not None:
                        loginfo["eval/simscore_en"] = simscore_en
                    if simscore_ja is not None:
                        loginfo["eval/simscore_ja"] = simscore_ja
                
                if args.dim == 2:
                    """---------------- visualize ----------------"""
                    if args.model.lower() == "fireword":
                        if args.lang in ["ja", "both"]:
                            fig = visualize_fire_ja(model, args.plot_words, tokenizer=viz_tokenizer)
                        else:
                            fig = visualize_fire(model, args.plot_words, tokenizer=viz_tokenizer)
                    else:
                        raise ValueError(args.model)
                    img = wandb.Image(_fig2array(fig))
                    plt.close(fig)
                    loginfo["wordfig"] = img
                wandb.log(loginfo)
                wandb.save(f"{best_savepath}/**", args.savedir, policy="end")

            model.train()

    model.eval()

    log_best = f"Best iteration: {best_iter}. Similarity score={best_simscore:.3g}"
    if args.lang == "both" and simscore_en is not None and simscore_ja is not None:
        log_best += f" (EN={simscore_en:.3g}, JA={simscore_ja:.3g})"
    log_best += f", loss={best_loss:.3g}, savepath={best_savepath}"
    logger.info(log_best)
    
    if args.use_wandb:
        wandb_log = {
            "eval/best_iter": best_iter,
            "eval/best_simscore": best_simscore,
            "eval/best_loss": best_loss,
            "eval/best_savepath": best_savepath,
        }
        if args.lang == "both":
            if simscore_en is not None:
                wandb_log["eval/best_simscore_en"] = simscore_en
            if simscore_ja is not None:
                wandb_log["eval/best_simscore_ja"] = simscore_ja
        wandb.log(wandb_log)


class SimpleVocab:
    def __init__(self, s2i, i2s, word_counts=None, unk='<unk>'):
        self.s2i = s2i
        self.i2s = i2s
        self.unk = unk
        self._word_counts = word_counts
        # Add special_name2i for compatibility with FireWord
        self.special_name2i = {unk: s2i.get(unk, 0)}
    
    def __len__(self):
        return len(self.s2i)
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.s2i.get(key, self.s2i.get(self.unk))
        elif isinstance(key, int):
            return self.i2s.get(key, self.unk)
        return None

    def counts_dict(self):
        if self._word_counts:
            # self._word_counts is {int_id: count}.
            # We need {str_word: count}.
            return {self.i2s[k]: v for k, v in self._word_counts.items() if k in self.i2s}
        return {word: 1.0 for word in self.s2i.keys()}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _fig2array(fig):
    with io.BytesIO() as buff:
        fig.savefig(buff, format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = data.reshape((int(h), int(w), -1))
    return img

from matplotlib.lines import Line2D
@torch.no_grad()
def visualize_fire(model: FireWord, words: List[str], r: float = 4, tokenizer=None):
    
    # Prepare data for plotting
    plot_data = [] # List of (positions, weights) tuples
    
    s2i = model.vocab.s2i
    unk = model.vocab.unk if hasattr(model.vocab, 'unk') else '<unk>'
    
    device = next(model.parameters()).device

    for w in words:
        if tokenizer:
            tokens = tokenizer(w)
        else:
            tokens = [w]
            
        # Filter tokens
        valid_tokens = []
        for t in tokens:
            if t in s2i:
                valid_tokens.append(t)
            else:
                if unk in s2i:
                    valid_tokens.append(unk)
        
        if not valid_tokens:
            logger.warning(f"Word '{w}' (tokens: {tokens}) has no valid tokens in vocab.")
            continue
            
        # Get measures
        ft = model[valid_tokens]
        measure = ft.measures
        pos = measure.get_x() # (num_tokens, n, dim)
        # ensure positions and weights live on the model device
        if isinstance(pos, torch.Tensor):
            pos = pos.to(device)

        if isinstance(measure.m, float):
            ws = torch.ones(pos.shape[0], pos.shape[1], dtype=pos.dtype, device=pos.device) * measure.m
        else:
            ws = measure.m.abs().to(device)
             
        # Flatten
        pos_flat = pos.reshape(-1, pos.shape[-1])
        ws_flat = ws.reshape(-1)
        
        plot_data.append((pos_flat, ws_flat))

    if not plot_data:
        logger.warning("No data to plot.")
        return plt.figure()

    # Calculate limits
    all_pos = torch.cat([p for p, w in plot_data], dim=0)
    
    xmax = max(r, all_pos[:, 0].max().item())
    xmin = min(-r, all_pos[:, 0].min().item())
    ymax = max(r, all_pos[:, 1].max().item())
    ymin = min(-r, all_pos[:, 1].min().item())

    # Ensure meshgrid is created on the same device as the model to avoid
    # cpu/cuda mixing when calling `model.field` below.
    device = next(model.parameters()).device
    xmesh = torch.linspace(xmin, xmax, 100, device=device)
    ymesh = torch.linspace(ymin, ymax, 100, device=device)
    xmesh, ymesh = torch.meshgrid(xmesh, ymesh)
    
    # Calculate field for the first word (phrase)
    score = torch.zeros_like(xmesh, device=device)
    w0 = words[0]
    if tokenizer:
        tokens0 = tokenizer(w0)
    else:
        tokens0 = [w0]
    
    valid_tokens0 = [t if t in s2i else unk for t in tokens0]
    valid_tokens0 = [t for t in valid_tokens0 if t in s2i]
    
    for t in valid_tokens0:
        # make sure arguments to model.field are on the model device
        tx, ty = xmesh, ymesh
        # if model.field expects CPU tensors for some reason, it will
        # handle device movement internally; usually it's faster to keep
        # everything on the same device
        score += model.field(t, tx, ty)

    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    colors = ["#370665", "#35589A", "#F14A16", "#FC9918"]
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    # Move data back to CPU for plotting
    xmesh, ymesh, score = list(map(lambda x: x.detach().cpu().numpy(), [xmesh, ymesh, score]))
    
    cont = ax.contourf(xmesh, ymesh, score)
    handlers = []
    
    for i, (pos, ws) in enumerate(plot_data):
        pos = pos.detach().cpu().numpy()
        ws = ws.detach().cpu().numpy()
        color = colors[i % len(colors)]
        
        h = None
        for (x, y), w in zip(pos, ws):
            (h,) = ax.plot(
                x,
                y,
                "o",
                color=color,
                markersize=_sigmoid(w) * 10,
                markeredgecolor="white",
            )
        if h:
            handlers.append(h)
            
    ax.legend(handlers, words[:len(handlers)])
    fig.colorbar(cont)
    return fig


@torch.no_grad()
def visualize_fire_ja(model: FireWord, words: List[str], r: float = 4, tokenizer=None):
    
    # Prepare data for plotting
    plot_data = [] # List of (word_label, list of (pos_flat, ws_flat, token_str))
    
    s2i = model.vocab.s2i
    unk = model.vocab.unk if hasattr(model.vocab, 'unk') else '<unk>'
    
    device = next(model.parameters()).device

    for w in words:
        if tokenizer:
            tokens = tokenizer(w)
        else:
            tokens = [w]
            
        # Filter tokens
        valid_tokens_info = []
        for t in tokens:
            if t in s2i:
                valid_tokens_info.append(t)
            else:
                if unk in s2i:
                    valid_tokens_info.append(unk)
        
        if not valid_tokens_info:
            logger.warning(f"Word '{w}' (tokens: {tokens}) has no valid tokens in vocab.")
            continue
            
        word_tokens_data = []
        for t in valid_tokens_info:
            ft = model[[t]]
            measure = ft.measures
            pos = measure.get_x() # (1, n, dim)
            if isinstance(pos, torch.Tensor):
                pos = pos.to(device)

            if isinstance(measure.m, float):
                ws = torch.ones(pos.shape[0], pos.shape[1], dtype=pos.dtype, device=pos.device) * measure.m
            else:
                ws = measure.m.abs().to(device)
             
            # Flatten
            pos_flat = pos.reshape(-1, pos.shape[-1])
            ws_flat = ws.reshape(-1)
            
            word_tokens_data.append((pos_flat, ws_flat, t))
        
        plot_data.append((w, word_tokens_data))

    if not plot_data:
        logger.warning("No data to plot.")
        return plt.figure()

    # Calculate limits
    all_pos_list = []
    for _, tokens_data in plot_data:
        for pos, _, _ in tokens_data:
            all_pos_list.append(pos)
            
    all_pos = torch.cat(all_pos_list, dim=0)
    
    xmax = max(r, all_pos[:, 0].max().item())
    xmin = min(-r, all_pos[:, 0].min().item())
    ymax = max(r, all_pos[:, 1].max().item())
    ymin = min(-r, all_pos[:, 1].min().item())

    # Ensure meshgrid is created on the same device as the model
    xmesh = torch.linspace(xmin, xmax, 100, device=device)
    ymesh = torch.linspace(ymin, ymax, 100, device=device)
    xmesh, ymesh = torch.meshgrid(xmesh, ymesh)
    
    # Calculate field for the first word (phrase)
    score = torch.zeros_like(xmesh, device=device)
    
    # Use the first word from plot_data
    first_word_tokens = plot_data[0][1]
    
    for _, _, t in first_word_tokens:
        tx, ty = xmesh, ymesh
        score += model.field(t, tx, ty)

    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    colors = ["#370665", "#35589A", "#F14A16", "#FC9918"]
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    # Move data back to CPU for plotting
    xmesh, ymesh, score = list(map(lambda x: x.detach().cpu().numpy(), [xmesh, ymesh, score]))
    
    cont = ax.contourf(xmesh, ymesh, score)
    handlers = []
    legend_labels = []
    
    for i, (word_label, tokens_data) in enumerate(plot_data):
        color = colors[i % len(colors)]
        
        h_word = None
        
        for j, (pos, ws, token_str) in enumerate(tokens_data):
            marker = markers[j % len(markers)]
            pos = pos.detach().cpu().numpy()
            ws = ws.detach().cpu().numpy()
            
            for (x, y), w in zip(pos, ws):
                (h,) = ax.plot(
                    x,
                    y,
                    marker,
                    color=color,
                    markersize=_sigmoid(w) * 10,
                    markeredgecolor="white",
                )
                if j == 0 and h_word is None:
                    h_word = h
        
        if h_word:
            handlers.append(h_word)
            legend_labels.append(word_label)
            
    ax.legend(handlers, legend_labels)
    fig.colorbar(cont)
    return fig


def parse_arguments():
    parser = argparse.ArgumentParser()

    def boolean_string(s):
        if s not in {"False", "True"}:
            raise ValueError("Not a valid boolean string")
        return s == "True"
    def parse_plot_words(s: str) -> List[str]:
        # "行く,重い,言葉" -> ["行く","重い","言葉"]
        # ついでに全角カンマも許容
        s = s.replace("，", ",")
        return [w.strip() for w in s.split(",") if w.strip()]


    # ----- experiment setting -----
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="data/corpus/text8",
    )
    parser.add_argument(
        "--vocab_path_json",
        type=str,
        default=None,
    )
    parser.add_argument("--model", type=str, default="FireWord", choices=["FireWord"])
    parser.add_argument("--task", type=str, default="skipgram", choices=["skipgram"])

    # ----- fire model settings -----
    parser.add_argument(
        "--dim",
        type=int,
        default=2,
        help="Dimension of time series. Only `dim=1` is supported now.",
    )
    parser.add_argument(
        "--func", type=str, default="MLP(args.dim, [8, 2, 8])", help="See parse_func()"
    )
    parser.add_argument("--measure", type=str, default="DiracMixture(args.dim, 1)")
    parser.add_argument(
        "--func_measure",
        type=str,
        default="",
        help="concat of `func` and `measure` with sep=@@. For example: `MLPlanarDiv(dim, 4)@@DiracMixture(dim, 10)`",
    )

    parser.add_argument("--grid_limits", type=str, default="[-4.0, 4.0]")
    parser.add_argument("--grid_dim_sizes", type=str, default="8")

    # ----- skipgram parameters -----
    parser.add_argument(
        "--win_size",
        type=int,
        default=10,
        help="Defines a neighborhood [x-win_size, x+win_size)",
    )
    parser.add_argument(
        "--unk",
        type=str,
        default="<unk>",
        help="Should be consistent with the vocabulary.",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=5,
        help="All words with fewer counts are regarded as {unk}",
    )
    parser.add_argument(
        "--read_mode",
        type=str,
        default="shuffle",
        choices=["shuffle", "repeat", "onepass"],
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1e-4,
        help="Down-sampling frequent words in positive sampling.",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=0.75,
        help="Up-sampling infrequent words in negative sampling.",
    )
    parser.add_argument(
        "--n_neg",
        type=int,
        default=1,
        help="number of negative samples per positive one",
    )

    # ----- optimize -----
    parser.add_argument("--n_iters", type=int, default=100000)
    parser.add_argument(
        "--sz_batch",
        type=int,
        default=8192,
        help="set it large to put all assets in one batch, or make it small to reduce memory usage",
    )
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--lr_scheduler", type=str, default="OneCycleLR")
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=10,
        help="Update parameters every several iterations.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=1000,
        help="model is evaluated every 20 iterations and the snapshot is saved.",
    )
    parser.add_argument(
        "--sinkhorn_weight",
        type=float,
        default=0.0,
        help="Weight of the Sinkhorn distance term in the total loss.",
    )
    parser.add_argument(
        "--sinkhorn_reg",
        type=float,
        default=1.0,
        help="Weight on the regularization term in the Sinkhorn distance",
    )
    parser.add_argument(
        "--sinkhorn_max_iter",
        type=int,
        default=50,
        help="A parameter of the Sinkhorn distance term that limits the number of estimating iterations.",
    )
    parser.add_argument(
        "--sinkhorn_p",
        type=float,
        default=2.0,
        help="Norm dimension of the Sinkhorn distance.",
    )
    parser.add_argument(
        "--sinkhorn_tau",
        type=float,
        default=1e3,
        help="Used for stablization of the Sinkhorn computation.",
    )
    parser.add_argument(
        "--sinkhorn_stop_threshold",
        type=float,
        default=1e-2,
        help="Controlling stop of the Sinkhorn iteration.",
    )

    # ----- miscellaneous -----
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="Use cpu rather than CUDA.")
    parser.add_argument("--savedir", type=str, default="./results/")
    parser.add_argument(
        "--plot_words",
        type=str,
        default="bank,river,ball",
        help='Comma-separated words, e.g. "行く,重い,言葉" or "bank,river,ball"',
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="*",
        help="List of (word) benchmarks for intermediate evaluation.",
    )
    parser.add_argument(
        "--benchmark_lower",
        type=boolean_string,
        default="True",
        help="convert benchmark texts to lower case.",
    )
    parser.add_argument(
        "--amp", action="store_true", help="Use half precision for accelleration."
    )
    parser.add_argument("--profile", action="store_true", help="CPU/GPU profiling.")

    # ----- wandb -----
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Save all results (and additional visualization results) to wandb.",
    )
    parser.add_argument(
        "--wandb_pretrained",
        type=str,
        default=None,
        help="Load a pre-trained FIRE from wandb, by its ID (e.g., allen/firelang/abcdefghi)",
    )
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--lang", type=str, default="en", choices=["en", "ja", "both"],
                        help="Language for benchmarking: 'en' (English only), 'ja' (Japanese only), or 'both'")

    args = parser.parse_args()
    args.plot_words = parse_plot_words(args.plot_words)

    if args.func_measure:
        args.func, args.measure = args.func_measure.split("@@")

    args.grid_limits = eval(args.grid_limits)
    args.grid_dim_sizes = eval(args.grid_dim_sizes)

    return args


if __name__ == "__main__":
    args = parse_arguments()
    train(args)
