# akkadian-mt

ByT5 を使って Deep Past Initiative - Machine Translation (Akkadian -> English) 向けに fine-tuning / inference を行う最小構成の実験基盤です。

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Data

学習データは CSV を想定します。以下のどちらかの列名セットに対応します。
- `source`, `target`
- Kaggle 形式の `transliteration`, `translation`

```csv
source,target
"a-na e2-gal ...","to the palace ..."
```

推論入力は以下のどちらかに対応します。
- `source`
- Kaggle `test.csv` の `transliteration`

例:
- `data/raw/kaggle/train.csv`
- `data/raw/kaggle/test.csv`

## Colab Example

```bash
!git clone <your-repo-url>
%cd akkadian-mt
!pip install uv
!uv venv
!uv pip install -r requirements.txt
!bash scripts/run_train.sh data/raw/kaggle/train.csv '' outputs/byt5-base
```

## Train

```bash
uv run python -m src.train \
  --config configs/byt5_base.yaml \
  --train_path data/raw/kaggle/train.csv \
  --output_dir outputs/byt5-base
```

`--valid_path` を省略すると、config の `validation_split_ratio` に従って train から split します。

## Inference

```bash
uv run python -m src.infer \
  --model_path outputs/byt5-base/best_checkpoint \
  --input_path data/raw/kaggle/test.csv \
  --output_path outputs/predictions.csv
```

## Outputs

`output_dir` 以下に主に次を保存します。
- `config.yaml`: 実行時設定
- `trainer_state.json`: Trainer 状態
- `all_results.json`: 最終 metrics
- `validation_predictions.csv`: valid 推論結果
- `checkpoint-*`: 中間 checkpoint
- `best_checkpoint/`: Kaggle notebook 推論へ持ち込みやすい最良 checkpoint のコピー

## Shell Scripts

```bash
bash scripts/run_train.sh data/raw/kaggle/train.csv '' outputs/byt5-base
bash scripts/run_infer.sh outputs/byt5-base/best_checkpoint data/raw/kaggle/test.csv outputs/predictions.csv
```
