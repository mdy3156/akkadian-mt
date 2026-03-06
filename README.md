# akkadian-mt

ByT5 を使って Deep Past Initiative - Machine Translation (Akkadian -> English) 向けに fine-tuning / inference を行う最小構成の実験基盤です。

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
- `data/processed/evacun/train.csv`

Evacun の `.txt` ペアから学習 CSV を作る例:

```bash
python scripts/build_evacun_csv.py \
  --transcription_path data/raw/kaggle/evacun/transcription_train.txt \
  --english_path data/raw/kaggle/evacun/english_train.txt \
  --output_path data/processed/evacun/train.csv
```

## Colab Example

```bash
!git clone <your-repo-url>
%cd akkadian-mt
!pip install -r requirements.txt
!python scripts/build_evacun_csv.py --transcription_path data/raw/kaggle/evacun/transcription_train.txt --english_path data/raw/kaggle/evacun/english_train.txt --output_path data/processed/evacun/train.csv
!bash scripts/run_train.sh data/raw/kaggle/train.csv '' outputs/byt5-base configs/byt5_base.yaml data/processed/evacun/train.csv
```

## Train

```bash
python -m src.train \
  --config configs/byt5_base.yaml \
  --train_path data/raw/kaggle/train.csv \
  --extra_train_paths data/processed/evacun/train.csv \
  --output_dir outputs/byt5-base
```

`--valid_path` を省略すると、最初の `--train_path` で与えた Kaggle train からだけ validation を split します。`--extra_train_paths` で追加した Evacun は train にのみ入ります。

## Inference

```bash
python -m src.infer \
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
bash scripts/run_train.sh data/raw/kaggle/train.csv '' outputs/byt5-base configs/byt5_base.yaml data/processed/evacun/train.csv
bash scripts/run_infer.sh outputs/byt5-base/best_checkpoint data/raw/kaggle/test.csv outputs/predictions.csv
```
