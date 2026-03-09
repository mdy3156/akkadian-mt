# akkadian-mt

公開済み Akkadian MT checkpoint を評価し、`data/train.csv` で fine-tune するための最小構成です。実行設定はすべて `configs/` の YAML で管理します。

## Scope

残している機能は 2 つだけです。
- 公開モデルの評価
- 公開モデルの fine-tune

学習データ変換や追加コーパス結合はこのリポジトリでは扱いません。

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Format

`data/train.csv`

```csv
source,target
a-na e2-gal ...,to the palace ...
```

- 列は `source,target` に固定です

## Configs

- [`configs/public_byt5_optimized.yaml`](/home/mdy/akkadian-mt/configs/public_byt5_optimized.yaml): fine-tune 用
- [`configs/eval_public.yaml`](/home/mdy/akkadian-mt/configs/eval_public.yaml): 公開モデル評価用

主なパラメータ:
- `model_path`
- `train_path` / `input_path`
- `output_dir`
- `sample_ratio`
- `seed`
- generation 設定

## Evaluate Public Model

固定 seed で `data/train.csv` の一部を切り出して、公開モデルの素の性能を見ます。

```bash
python -m src.evaluate_checkpoint --config configs/eval_public.yaml
```

保存物:
- `eval_metrics.json`
- `eval_predictions.csv`

`eval_predictions.csv` には `source_raw`, `source_preprocessed`, `prediction`, `target` を保存します。識別用の `id` は内部で連番を振ります。

## Fine-tune

```bash
python -m src.train --config configs/public_byt5_optimized.yaml
```

別の公開モデルや出力先を使う場合も、CLI 引数ではなく config を編集します。
学習時は `train_path` を seed 固定で train/validation に分割します。
