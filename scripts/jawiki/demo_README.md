# 日本語処理デモ

このディレクトリには、日本語コーパスの処理における **品詞フィルタ** の効果を確認するためのデモスクリプトが含まれています。

## 概要

| スクリプト | 説明 |
|-----------|------|
| `demo_2_tokenize.sh` | トークン化（全形態素 vs 品詞フィルタ） |
| `demo_3_build_vocab.sh` | 語彙構築（結果の比較） |
| `demo_4_train.sh` | 学習（インタラクティブに選択可能） |

## 品詞フィルタとは

MeCabで形態素解析を行う際に、**特定の品詞のみ** を抽出するフィルタです。

### 抽出対象の品詞
- 動詞
- 名詞
- 形容詞
- 形容動詞
- 副詞
- 助動詞

### 除外される品詞
- 助詞（は、が、を、に、で、など）
- 接続詞
- 記号・句読点
- 感動詞
- など

## 例

```
入力文: 私は東京に行きました。

【通常版トークン化】
私 は 東京 に 行き まし た 。

【品詞フィルタ版トークン化】
私 東京 行き まし た
```

## 実行方法

### 1. トークン化
```bash
cd /home/maruyama/projectB/firelang
bash scripts/jawiki/demo_2_tokenize.sh
```

### 2. 語彙構築
```bash
bash scripts/jawiki/demo_3_build_vocab.sh
```

### 3. 学習
```bash
bash scripts/jawiki/demo_4_train.sh
```

## 期待される効果

| 項目 | 通常版 | 品詞フィルタ版 |
|------|--------|----------------|
| 語彙サイズ | 大きい | 小さい（20-30%減の可能性） |
| トークン数 | 多い | 少ない（30-40%減の可能性） |
| 学習速度 | 標準 | やや速い |
| 意味類似度タスク | 標準 | 向上の可能性 |
| OOV問題 | 発生しやすい | 軽減される |

## ファイル構成

```
data/corpus/ja-text8/
├── ja-text8                           # 元のコーパス
├── ja-text8.uncased.tokens            # 通常版トークン
├── ja-text8.uncased.tokens.vocab.json # 通常版語彙
├── ja-text8.uncased.tokens.word_counts.json
├── ja-text8.uncased.filtered.tokens   # フィルタ版トークン
├── ja-text8.uncased.filtered.tokens.vocab.json
└── ja-text8.uncased.filtered.tokens.word_counts.json
```

## 注意事項

1. **MeCabが必要です**
   ```bash
   pip install mecab-python3 unidic-lite
   ```

2. **コーパスが必要です**
   先に `1_download_ja-text8.sh` を実行してコーパスをダウンロードしてください。

3. **GPU環境推奨**
   学習はGPUがあると高速です。CPUでも動作しますが時間がかかります。

## トラブルシューティング

### MeCabエラーが出る場合
```bash
pip install --upgrade mecab-python3 unidic-lite
```

### CUDA/PTXエラーが出る場合
GPUドライバとCUDAツールキットのバージョン不整合の可能性があります。
CPU実行に切り替える場合:
```bash
CUDA_VISIBLE_DEVICES="" bash scripts/jawiki/demo_4_train.sh
```
