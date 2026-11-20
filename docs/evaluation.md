# ANN training and evaluation

## Reproduce training (PowerShell)

From the repository root:

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = (Get-Location).Path
python -m app.ml.train --dataset data/ann/enterprise_it_statement_pairs.jsonl --artifact-root artifacts/ann --seed 42 --embedding-dim 48 --hidden-dims 128,64 --dropout 0.20 --learning-rate 0.003 --batch-size 16 --epochs 120 --patience 18
```

The command writes a versioned folder such as `artifacts/ann/ann-v1-seed42/`. It includes the model weights, vocabulary, training-only scaler, baseline artifacts, training history, metrics, metadata, and manifest. It does not access Pinecone or OpenAI.

## Dataset and split

The bundled JSONL has 90 engineer-curated seed statement pairs across 15 source-document groups, with two examples per class (`CONTRADICTION`, `CONSISTENT`, `UNCERTAIN`) per group. The split is deterministic (seed 42 by default) and group-disjoint: 10 groups for training, 2 for validation, and 3 for test (60/12/18 pairs). This reduces cross-split leakage from related statements in the same policy topic. All learned preprocessing is fitted on training rows only.

**Important evaluation caveat:** these labels are not independently human-reviewed. The resulting test metrics are pipeline/development diagnostics, not valid evidence of real-world enterprise performance. The specification's human-reviewed domain test-set acceptance criterion remains open. Do not make production accuracy claims from this seed dataset.

## Metrics

`evaluation_report.json` records train/validation/test group and label counts, epoch-wise train/validation losses, best epoch, validation-selected low-confidence threshold, confusion matrices, accuracy, macro-F1, and per-class precision/recall/F1 for both the ANN and logistic-regression baseline. Both raw and thresholded ANN predictions are reported. Values must be read from the generated report; no performance value is assumed by this document.

## Inference smoke test

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -c "from app.ml.inference import load_predictor; p=load_predictor('artifacts/ann/ann-v1-seed42'); print(p.predict('MFA is required for all administrators.', 'Administrators must complete multi-factor authentication at login.'))"
```

A low-confidence result is converted to `UNCERTAIN` and marked `review_required`. The raw `model_label` is preserved separately from `predicted_label`.

## Test the implementation

```powershell
python -m unittest discover -s tests -v
```

## Limitations

The sentence encoder is a task-trained mean-pooled word-embedding model, not a pretrained general-purpose transformer. It is separate from the OpenAI embedding model used by Pinecone. A larger independently human-reviewed test set and external review are required before deployment or performance claims.
