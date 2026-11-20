# Model card: Enterprise IT statement relationship ANN

## Intended use

A supervised research/development classifier for pairs of statements extracted from different enterprise IT documents. It predicts one of `CONTRADICTION`, `CONSISTENT`, or `UNCERTAIN`. The output is evidence triage, not a determination of truth, policy precedence, legal authority, or organizational authority. A low-confidence result is surfaced as `UNCERTAIN` and requires human review.

## Model architecture

- **Classifier:** PyTorch feed-forward multilayer perceptron (MLP).
- **Sentence encoder:** task-trained mean-pooled word embeddings. Each token is mapped to a learned 48-dimensional vector (configurable); token vectors are mean-pooled and L2-normalized. The sentence encoder and MLP are trained jointly on the training split.
- **Pair features:** sentence embedding A, sentence embedding B, element-wise absolute difference, element-wise product, cosine similarity, numeric/date mismatch, negation mismatch, token Jaccard overlap, and statement-length ratio.
- **Structured preprocessing:** `StandardScaler` is fitted only on training rows and saved with the model.
- **Label order:** `CONTRADICTION`, `CONSISTENT`, `UNCERTAIN`.
- **Separate embedding space:** this model's learned word embeddings are independent of the existing OpenAI/Pinecone document embedding configuration. They are not substituted into the RAG index.

This is a small task-specific sentence encoder, not a pretrained general-purpose semantic encoder. It has no external model download or API dependency. The vocabulary is learned from training statements only; unseen words map to `<UNK>`.

## Training data

`data/ann/enterprise_it_statement_pairs.jsonl` contains 90 engineer-curated seed pairs across 15 document groups. It covers VPN, MFA, passwords, account lockout, endpoint devices, patching, software approvals, remote access, access control, backup, email, administrator privileges, data handling, browsers, and incident reporting. The JSONL schema includes `statement_a`, `statement_b`, `label`, `source_doc_a`, `source_doc_b`, `rationale`, `group_id`, and `label_status`.

The current labels are **not independently human-reviewed**. They are suitable for exercising the training pipeline and demonstrating reproducibility, not for asserting enterprise-domain validity. No public NLI dataset is included or used.

## Split and leakage controls

A seeded split assigns entire `group_id` source-document groups to train, validation, or test. The vocabulary, structured-feature scaler, and baseline TF-IDF vocabulary are fitted on training rows only. Validation is used for early stopping and confidence-threshold selection. The held-out test split is used once for the reported development diagnostics.

## Baseline

A scikit-learn logistic-regression classifier is trained on TF-IDF pair features and the same four structured features. Its vectorizer and classifier are retained alongside the ANN for a like-for-like diagnostic comparison.

## Prediction contract

`app.ml.inference.load_predictor(artifact_dir).predict(statement_a, statement_b)` returns:

- `predicted_label` and unthresholded `model_label`;
- per-class `probabilities` and maximum `confidence`;
- `model_version`, `threshold`, and `review_required`;
- sentence encoder identity and a warning that this is not proof of factual truth.

When maximum class probability is below the validation-selected threshold, `predicted_label` is `UNCERTAIN` and `review_required` is true. An unthresholded `UNCERTAIN` prediction also requires review.

## Limitations and required next steps

1. The current seed labels are not human-reviewed. Obtain an independently adjudicated enterprise-domain dataset before making quality claims.
2. The small word vocabulary and task-trained mean pooling may not generalize to unseen IT terminology, paraphrases, or long context.
3. Test metrics on this curated seed set are development diagnostics only; do not present them as production accuracy or evidence of business impact.
4. Expand examples across organizations, document authors, policy versions, and writing styles; measure inter-annotator agreement and document label adjudication.
5. Validate calibration on a larger independent validation set and conduct human-in-the-loop review before deployment.

## Artifact contents

Each versioned artifact directory contains `model.pt`, `model_metadata.json`, `vocabulary.json`, `scalar_scaler.joblib`, baseline vectorizer/classifier, training history, evaluation report, and a SHA-256 manifest. Generated artifacts are ignored by Git by default.
