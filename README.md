# akkadian-mt

公開済み Akkadian MT checkpoint または Hugging Face model ID を評価し、`data/train.csv` で fine-tune するための最小構成です。実行設定はすべて `configs/` の YAML で管理します。

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

`data/eval.csv`

```csv
source,target
a-na e2-gal ...,to the palace ...
```

- 任意です
- `configs/train.yaml` で `eval_path` を指定すると、学習中 evaluation はこのファイルを使います
- `eval_path` が無い場合だけ、`train_path` から `validation_split_ratio` で分割します

## Configs

- [`configs/train.yaml`](/home/mdy/akkadian-mt/configs/train.yaml): fine-tune 用
- [`configs/eval.yaml`](/home/mdy/akkadian-mt/configs/eval.yaml): 公開モデル評価用
- [`configs/train_byt5_large.yaml`](/home/mdy/akkadian-mt/configs/train_byt5_large.yaml): `google/byt5-large` fine-tune 用
- [`configs/eval_byt5_large.yaml`](/home/mdy/akkadian-mt/configs/eval_byt5_large.yaml): `google/byt5-large` 評価用

主なパラメータ:
- `model_path`
- `train_path` / `input_path`
- `output_dir`
- `sample_ratio`
- `seed`
- generation 設定
- `bidirectional_augmentation`

`model_path` には次のどちらも指定できます。
- ローカル checkpoint ディレクトリ
- Hugging Face model ID 例: `google/byt5-large`

`bidirectional_augmentation: true` にすると、train データだけ次の 2 方向へ増やします。
- `translate Akkadian to English: source -> target`
- `translate English to Akkadian: target -> source`

validation / evaluation は従来どおり Akkadian -> English のみです。

## Evaluate Public Model

固定 seed で `data/train.csv` の一部を切り出して、公開モデルの素の性能を見ます。

```bash
python -m src.evaluate_checkpoint --config configs/eval.yaml
```

`google/byt5-large` を直接評価する場合:

```bash
python -m src.evaluate_checkpoint --config configs/eval_byt5_large.yaml
```

保存物:
- `eval_metrics.json`
- `eval_predictions.csv`

`eval_predictions.csv` には `source_raw`, `source_preprocessed`, `prediction`, `target` を保存します。識別用の `id` は内部で連番を振ります。

## Fine-tune

```bash
python -m src.train --config configs/train.yaml
```

別の公開モデルや出力先を使う場合も、CLI 引数ではなく config を編集します。
学習時は `eval_path` があればそれを評価に使い、無ければ `train_path` を seed 固定で train/validation に分割します。

`google/byt5-large` を直接 fine-tune する場合:

```bash
python -m src.train --config configs/train_byt5_large.yaml
```

`byt5-large` は重いので、まずは次を前提にしてください。
- A100 以上
- `bf16: true`
- `per_device_train_batch_size: 1`
