# Dataset Authenticity and Benchmark Split Integrity Report

- **Overall Audit Status:** PASSED
- **Execution Mode:** `final`
- **Allow Synthetic:** `False`
- **Experiment Seed:** `42`
- **Partition Policy:** Strict Benchmark Preservation (Train-only calibration/validation, zero eval truncation)

## 1. Task Benchmark & Partition Summary

| Task | Dataset (Config) | Resolved Git Commit SHA | Original Sources | Excluded Dups | Train | Calib | Val | Eval | Integrity |
|---|---|---|---|---|---|---|---|---|---|
| boolq | google/boolq (default) | `35b264d036` | train:9427, evaluation:3270 | 0 | 9199 | 128 | 100 | 3270 | **PASSED** |
| gsm8k | openai/gsm8k (main) | `740312add8` | train:7473, evaluation:1319 | 0 | 7245 | 128 | 100 | 1319 | **PASSED** |
| arc_easy | allenai/ai2_arc (ARC-Easy) | `210d026faf` | train:2251, validation:570, evaluation:2376 | 2 | 2121 | 128 | 570 | 2376 | **PASSED** |

## 2. Upstream Cross-Split Duplicate Exclusions

The following upstream duplicates were detected in training sources colliding with official benchmark evaluation/validation splits. In accordance with benchmark preservation protocol, official evaluation examples were kept intact and training duplicates were excluded from training/calibration pools:

### Task: `arc_easy` (2 duplicate(s) excluded)

- **Train Index 1087** (`id`: `CSZ_2009_8_CSZ30768`) collides with `test` index 242 (`id`: `CSZ30768`):
  - Question Excerpt: *"Which of the following elements is best able to combine with itself and hydrogen [H] to form large molecules?"*
  - Conflicting Target: `False` (Train: `D`, Eval: `D`)

- **Train Index 2246** (`id`: `CSZ_2009_8_CSZ20770`) collides with `test` index 493 (`id`: `CSZ20770`):
  - Question Excerpt: *"Iron oxides, such as rust, form when iron metal reacts with oxygen in the air. What are the chemical symbols for the two"*
  - Conflicting Target: `False` (Train: `C`, Eval: `C`)

## 3. Cryptographic Partition Content Fingerprints (SHA-256)

### Task: `boolq`

| Split | Number of Examples | Content SHA-256 Fingerprint |
|---|---|---|
| train | 9199 | `4c73b06d546e0823601c4841924c1d7ff04020518816f1433c87fa4ff653c407` |
| calibration | 128 | `e122507e189a02fd00cc23e363ee39e842c765f21a74564af1d938a1bd273003` |
| validation | 100 | `f537e8f6d6abe7b24bb48e8bf1a229a89438c9b93aeff2f3d042d4522087cdd0` |
| evaluation | 3270 | `b814b6b952bf050a1cacdaf039b89316f3758ca5fd9f86b6d533d26b2afaa106` |

### Task: `gsm8k`

| Split | Number of Examples | Content SHA-256 Fingerprint |
|---|---|---|
| train | 7245 | `40dfd238d68d973145d97da1ddbf1c8287cab63f1a4d35963b85f1650983802b` |
| calibration | 128 | `29e626e3b28b6a8edff3b2700e3525eadd30a882c389516bee4c95fdc0ad9e9c` |
| validation | 100 | `2bb0b261cf1e35ceaa1c6a509db43b96e6fb6a7f8f79b462415402145d855c1f` |
| evaluation | 1319 | `cc606eb92fba021c244fa7c269b5f876bb0b1b235324031e745ae95947facab0` |

### Task: `arc_easy`

| Split | Number of Examples | Content SHA-256 Fingerprint |
|---|---|---|
| train | 2121 | `dc749131fa06af1d1b064a5f495e7eff83197a1a8ca5c72dff9aff529252a8ef` |
| calibration | 128 | `202163cb3c0d02937caf774841845af5ae3670a7d5b73efdafbd548052d24d15` |
| validation | 570 | `23aca3f257b4f965b988ed8be47c2f802c3b91b3c046686bfe13567ee660b475` |
| evaluation | 2376 | `33ffc4961089d5c8a68e850992388016582b1b1dee54eaa7a7e161ec13f0700c` |

