# Large Language Model-Based Pedagogical Profile Generation for Cold-Start Online Course Recommendation

This repository contains the data processing, course-profile generation, recommendation experiments, and saved per-user scores for the paper. The reported method named **MVP** in the manuscript is implemented here as **SemProj**.

The experiments ask whether pedagogical profiles improve cold-start recommendation (RQ1), whether gains vary between new and near-duplicate courses (RQ2), whether the five profile fields contribute to performance (RQ3), and what profile-generation cost and inference latency they add (RQ4).

## Repository contents

| Path | Purpose |
|---|---|
| `data/` | Source-derived tables, fixed train/validation/test splits, course profiles, text vectors, and LightGCN embeddings |
| `llm/` | Qwen-based course-profile generation and Sentence-BERT encoding |
| `src/data/` | Data preparation and split notebooks |
| `src/models/` | Cold-start and warm-scenario model notebooks and Python modules |
| `src/eval/` | Evaluation utilities, cold-item checks, and supplementary analyses |
| `scores/` | Per-user NDCG@20 arrays used for the published statistical comparisons |
| `results/` | Statistical tables, verdicts, supplementary outputs, and the final-statistics notebook |
| `archive/` | Few-shot-user data and exploratory notebooks not used in the paper |

All notebooks locate the repository root automatically. When running code from another working directory, set `DATA_DIR` to the repository root:

```powershell
$env:DATA_DIR = (Get-Location).Path
```

```bash
export DATA_DIR="$PWD"
```

## Data and environment

The source dataset is [THU-KEG/MOOCCubeX](https://github.com/THU-KEG/MOOCCubeX). The upstream repository includes a [GPL-3.0 license file](https://github.com/THU-KEG/MOOCCubeX/blob/main/LICENSE), while the [MOOCCubeX paper](https://lfs.aminer.cn/misc/moocdata/publications/mooccubex.pdf) describes data licensed by XuetangX; the reviewed materials do not state separate data-only terms. Preserve upstream attribution and check the applicable data terms before redistributing the data. This repository has no root `LICENSE` file, so it does not declare an overall license for its code.

The checkout includes the processed inputs and derived artifacts used in the reported experiments. The large interaction files are stored with Git LFS. Install the captured Python dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
git lfs install
git lfs pull
```

The recorded environment is Python 3.13.15 and PyTorch 2.11.0+cu128. The original GPU model was not recorded. CUDA is required for profile generation and GPU latency measurements; data validation and statistical analysis can run on CPU. The pinned PyTorch build targets CUDA 12.8.

The split preparation samples 100,000 learners with seed 2026 and does not apply a k-core filter. The committed cold-item lists select the latest 20% of courses by first recorded enrollment time: 753 items total, with 251 for validation and 502 for test. `check_cold_items.py` verifies the saved lists against that selection rule.

## Reproduction

Run commands from the repository root after installing dependencies and retrieving Git LFS files.

### 1. Quick demo and inference

Check that the saved cold-item lists agree with the latest-course selection rule, then generate one course profile as a small LLM inference example:

```bash
python src/eval/check_cold_items.py --strict
python llm/course_profile.py --num_courses 1 --max_new_tokens 300 --output_dir results/quick_demo
```

The inference command downloads Qwen2.5-7B-Instruct and the Sentence-BERT encoder on first use. It requires access to the model repository and a compatible NVIDIA GPU. Demo outputs are written to `results/quick_demo/`.

### 2. Evaluate the reported results

The repository contains LightGCN embeddings and all per-user score arrays used for the paper's tables. It does **not** contain trained weight checkpoints for every recommendation model. The evaluation below recomputes the statistical tables from the saved scores; regenerate scores by following the training workflow if you need to rerun model inference.

```bash
python -c "from pathlib import Path; Path('results/reproduction_runs').mkdir(parents=True, exist_ok=True)"
python src/eval/check_cold_items.py --strict
python src/eval/random_ndcg_analytic.py
python src/eval/time_consistent_subset.py
python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output final-stats-executed.ipynb --output-dir results/reproduction_runs results/final-stats.ipynb
```

The executed notebook is saved under `results/reproduction_runs/`. It recreates `results/final_stats.csv`, `results/final_stats_tables_with_notes.csv`, and `results/verdicts.json` from the checked-in score arrays.

| Paper result | Notebook or script | Main output | Expected check |
|---|---|---|---|
| Table 1: data splits | `src/data/02-data-splits.ipynb` | `data/splits/cold_scenario/`, `data/splits/warm_scenario/`, `results/split_statistics.csv` | 100,000 sampled learners; 753 cold items (251 validation, 502 test) |
| Course profiles and embeddings | `llm/course_profile.py` | `data/course_llm/2_profiles_full_and_shards/`, `data/course_llm/1_embeddings/` | 3,769 courses; 384-dimensional text vectors |
| Masked-prerequisite evaluation | `llm/course_profile.py` (`--masked_prereq_eval` option) | `data/course_llm/3_quality_evaluation/masked_prereq_200_eval_report.json` | 200-course report |
| Leakage sanity check | `src/models/lightgcn_leakage.ipynb` | Notebook output | Warm/cold split assertions complete without leakage flags |
| LightGCN backbone | `src/models/cold_scenario/lightgcn_backbone_cold.ipynb` | `data/embeddings/01_lightgcn_backbone_cold/g_u.npy`, `g_i.npy` | MD5 prefixes `7a170558` and `d17ad9df` |
| Random and ID baselines | `src/eval/evaluate_test_random.ipynb`, `src/models/cold_scenario/lightgcn_id_cold.ipynb` | Notebook output; `results/random_ndcg_analytic.txt` | Empirical random cold-test NDCG@20: 0.0157 ± 0.0010; LightGCN ID reference: 0.0165 |
| Content-KNN | `src/models/cold_scenario/content-knn-v2.ipynb` | `scores/Content-KNN/` | Raw+Profile versus Raw ΔNDCG@20 ≈ +0.0001 |
| DropoutNet | `src/models/cold_scenario/dropoutnet-v2.ipynb`, `dropoutnet-v2-p-rq3.ipynb`, `dropoutnet-llm-3d.ipynb` | `scores/DropoutNet/` | RQ1 Δ=+0.0103; new-course RQ2 Δ=+0.0147; five-field RQ3 Δ=+0.0044 |
| ALDI | `src/models/run_aldi.py` and `src/models/coldrec-adapter.ipynb` | `scores/ALDI/` | RQ1 Δ=−0.0018; not significant in the saved comparison |
| MVP (SemProj in this repository) | `src/models/semproj_experiments.ipynb` | `scores/SemProj/` | RQ1 Δ=+0.0142; new-course RQ2 Δ=+0.0212; five-field RQ3 Δ=+0.0122 |
| RQ4: profile-generation cost and latency | `llm/course_profile.py`, `src/models/semproj_experiments.ipynb`, `src/models/cold_scenario/content-knn-v2.ipynb` | Notebook output; `results_semproj/` | 12.13 GPU-hours; 11.58 s/course; peak VRAM about 5.5 GB; Content-KNN latency 1.49×/2.22× |
| Tables 2–4 and RQ verdicts | `results/final-stats.ipynb` | `results/final_stats.csv`, `results/verdicts.json` | RQ1 architecture-dependent; RQ2 concentrated on genuinely new courses; RQ3 contributes |
| Supplementary analyses | `src/eval/random_ndcg_analytic.py`, `src/eval/time_consistent_subset.py` | `results/random_ndcg_analytic.txt`, `results/time_consistent_subset.txt` | Random-ranking expectations: 0.01587648 (cold-only) and 0.00211462 (full catalog); time-consistent subset: 1,402/5,531 interactions |

The cold-item lists committed under `data/splits/cold_scenario/` are the canonical lists for reproducing the paper. To generate a new deterministic 80/20 cold-item selection and save it separately, run:

```bash
python src/eval/generate_cold_items.py --seed 2026
python src/eval/check_cold_items.py --lists-dir data/splits/cold_scenario/generated_cold_items --strict
```

The generated validation/test assignment is for a new run and does not replace the committed paper split. Keep the committed JSON files when reproducing the published scores.

### 3. Train from scratch

The repository contains the source-derived data required by the preparation notebooks. Run the preparation steps in order. The following commands save executed copies under `results/reproduction_runs/`:

```bash
python -c "from pathlib import Path; Path('results/reproduction_runs').mkdir(parents=True, exist_ok=True)"
python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output combine-data-executed.ipynb --output-dir results/reproduction_runs src/data/01-combine-data.ipynb
python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output data-splits-executed.ipynb --output-dir results/reproduction_runs src/data/02-data-splits.ipynb
```

Generate course profiles in two shards, then merge them and encode the course text:

```bash
python llm/course_profile.py --num_courses 0 --shard 0 --num_shards 2 --output_dir data/course_llm/2_profiles_full_and_shards
python llm/course_profile.py --num_courses 0 --shard 1 --num_shards 2 --output_dir data/course_llm/2_profiles_full_and_shards
python llm/course_profile.py --merge_shards --num_shards 2 --output_dir data/course_llm/2_profiles_full_and_shards
```

Train the backbone and recommendation models. These runs can take hours and require a CUDA-capable GPU for the neural models.

```bash
python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output lightgcn-leakage-executed.ipynb --output-dir results/reproduction_runs src/models/lightgcn_leakage.ipynb
python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output lightgcn-backbone-executed.ipynb --output-dir results/reproduction_runs src/models/cold_scenario/lightgcn_backbone_cold.ipynb
python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output content-knn-executed.ipynb --output-dir results/reproduction_runs src/models/cold_scenario/content-knn-v2.ipynb
python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output dropoutnet-executed.ipynb --output-dir results/reproduction_runs src/models/cold_scenario/dropoutnet-v2.ipynb
python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output dropoutnet-rq3-executed.ipynb --output-dir results/reproduction_runs src/models/cold_scenario/dropoutnet-v2-p-rq3.ipynb
python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output dropoutnet-3d-executed.ipynb --output-dir results/reproduction_runs src/models/cold_scenario/dropoutnet-llm-3d.ipynb
python src/models/run_aldi.py
python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output semproj-executed.ipynb --output-dir results/reproduction_runs src/models/semproj_experiments.ipynb
python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output final-stats-executed.ipynb --output-dir results/reproduction_runs results/final-stats.ipynb
```

The optional raw-only USIM screen is a limitation analysis and is not included in the paper's result tables. CGRC is not part of this workflow. To repeat the USIM screen, run:

```bash
python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output usim-raw-executed.ipynb --output-dir results/reproduction_runs src/models/coldrec-adapter-usim-raw.ipynb
```

## External implementations and citation

The optional ColdRec/USIM screen uses upstream ColdRec commit `1108192665aca2cb94b3aeb5bfed29850874eb1c` (MIT). ALDI is implemented in the repository's local PyTorch runner; its upstream reference is [zfnWong/ALDI](https://github.com/zfnWong/ALDI), commit `87f9fe369427402bd989728c4d27b543b2d19cd4`. Review upstream license terms before redistributing third-party code.

Please cite MOOCCubeX when using the dataset:

```bibtex
@inproceedings{yu2021mooccubex,
  title={MOOCCubeX: A large knowledge-centered repository for adaptive learning in MOOCs},
  author={Yu, Jifan and Wang, Yuquan and Zhong, Qingyang and Luo, Gan and Mao, Yiming and Sun, Kai and Feng, Wenzheng and Xu, Wei and Cao, Shulin and Zeng, Kaisheng and others},
  booktitle={Proceedings of the 30th ACM International Conference on Information & Knowledge Management},
  pages={4643--4652},
  year={2021}
}
```

The paper's author, venue, year, and DOI metadata were not included in this repository snapshot. Cite the final published paper metadata when available.
