# Large Language Model-Based Pedagogical Profile Generation for Cold-Start Online Course Recommendation

This repository contains the code, processed data, model scores, and analysis notebooks for the paper named above. It studies whether LLM-generated pedagogical profiles improve cold-start online course recommendation.

The experiments address four questions:

- **RQ1:** Does adding LLM-generated pedagogical profiles to raw course texts improve cold-start course recommendation performance across different model architectures?
- **RQ2:** Are the performance gains from pedagogical profiles concentrated on genuinely new courses or near-duplicate courses?
- **RQ3:** Do specific pedagogical fields (e.g., prerequisite knowledge and follow-up topics) contribute meaningfully to recommendation performance?
- **RQ4:** What are the computational costs of LLM-based profile generation and the inference latency overhead during recommendation?

## 1. Overview

The repository combines LLM-based course profile generation, collaborative filtering models, and an evaluation workflow for cold-start online course recommendation.

## 2. Repository structure

```text
.
├── archive/                 # Few-shot data and exploratory notebooks not used in the paper
├── data/                    # Source-derived data, splits, profiles, and saved embeddings
├── llm/                     # Course-profile generation, text processing, and SBERT encoding
├── results/                 # Final statistical tables, verdicts, and appendix reports
├── scores/                  # Per-user recommendation scores for model comparisons
└── src/
    ├── data/                # Data preparation and splitting notebooks
    ├── eval/                # Evaluation, statistical analysis, and verification scripts
    ├── losses/              # Auxiliary training losses
    └── models/              # Model modules and cold/warm scenario notebooks
        ├── cold_scenario/
        └── warm_scenario/
```

`archive/` holds the few-shot-user scenario data and four exploratory notebooks that are not used in the paper. The warm scenario files remain under `data/splits/warm_scenario/`, `data/embeddings/02_lightgcn_backbone_warm/`, and `src/models/warm_scenario/`; they support the leakage sanity check in Section 4.2.

## 3. Environment

Saved notebook metadata records **Python 3.13.15**. Executed model cells report **PyTorch 2.11.0+cu128**, CUDA available, and the CUDA 12.8 wheel build. The GPU model was not recorded in the saved notebook outputs. These are the captured experiment details; the full `pip freeze` from the original hosted runtime was not preserved, so the remaining pins in `requirements.txt` are a reproducible dependency set rather than a recovered full environment lock.

## 4. Installation

From the repository root:

```bash
python -m pip install -r requirements.txt
git lfs pull
```

The notebook outputs and model weights use Git LFS. GPU execution is needed for LLM generation and the GPU latency measurements; data checks and statistical scripts can run on CPU.

## 5. Data

The source is [THU-KEG/MOOCCubeX](https://github.com/THU-KEG/MOOCCubeX). Its GitHub repository is marked GPL-3.0, which permits redistribution of material covered by that license when its conditions are followed. The MOOCCubeX paper says the MOOC data was licensed by XuetangX and that the authors were permitted to maintain updates; the upstream materials reviewed do not identify a separate data-only license. The processed files are retained in this checkout with upstream attribution; confirm the scope of the data permission and preserve applicable notices before publishing a raw-data mirror.

Key paths:

- `data/raw_data_processed/`: processed MOOCCubeX CSV files. The large `user-course.csv` and `interactions_full.parquet` files are tracked by Git LFS.
- `data/full_interactions/interactions_full.parquet`: deduplicated interaction history used for splitting and evaluation.
- `data/course_llm/`: combined course table, generated profiles, quality checks, and text embeddings.
- `data/splits/cold_scenario/`: train, validation, and test data plus fixed cold-item ID lists.
- `data/splits/warm_scenario/`: warm split used in the Section 4.2 sanity check.
- `data/embeddings/01_lightgcn_backbone_cold/` and `02_lightgcn_backbone_warm/`: saved LightGCN user/item embeddings and index maps.
- `archive/few_shot_user_scenario/`: moved few-shot-user split files, kept for reference and excluded from the paper's main experiments.

The interaction preparation notebook samples **100,000 users with seed 2026** and does not apply a k-core filter. Cold items are selected from the latest 20% of courses by first observed enrollment time. The saved test list contains 502 items and the validation list 251, for 753 cold items in total. `src/eval/check_cold_items.py` recomputes the selection from `interactions_full.parquet`; the check passed with a 753/753 exact match and disjoint validation/test lists. The two JSON files are fixed artifacts produced by the earlier split run.

## 6. Pipeline

Run the steps from the repository root in order:

1. Combine raw tables: `src/data/01-combine-data.ipynb`.
2. Build the cold and warm splits: `src/data/02-data-splits.ipynb`.
3. Generate course profiles: `python llm/course_profile.py` with the desired shard options.
4. Encode raw descriptions and generated profiles with SBERT: `python llm/course_profile.py --encode_raw_only` for raw text; profile merging also writes the profile embeddings.
5. Train the LightGCN backbone: `src/models/cold_scenario/lightgcn_backbone_cold.ipynb` (and the warm notebook for the Section 4.2 check).
6. Run the four main cold-start models: Content-KNN, DropoutNet, ALDI, and SemProj using the notebooks in `src/models/cold_scenario/`, `src/models/coldrec-adapter.ipynb`, and `src/models/semproj_experiments.ipynb`.
7. Rebuild the statistical comparisons: `results/final-stats.ipynb`.

The course-profile CLI defaults to the local course table and prompt. For example:

```bash
python llm/course_profile.py --data_path data/course_llm/course_data.parquet --prompt_path src/prompts/prompt_template.txt --encode_raw_only --output_dir data/course_llm
```

Profile generation requires access to the selected Hugging Face model and a compatible GPU. The saved profile, embedding, backbone, and score artifacts let readers inspect the reported results without rerunning training.

### How to run `reproduce_paper.ipynb`

The notebook is a standalone file in the repository; the README provides instructions and a link to it. Open [reproduce_paper.ipynb](reproduce_paper.ipynb), then run it in Kaggle, Google Colab, or Jupyter/VS Code on a local machine. When sharing the project, commit the notebook and the related source changes to the repository. Readers can open or download the notebook from this link.

1. **Prepare the environment.** Enable Internet access so the notebook can fetch the source code, Git LFS data, and Python packages. Ensure that the machine has enough disk space and memory for the data and experiments. By default, the notebook uses the saved profiles and embeddings, so a GPU is not needed to regenerate them. Regenerating Qwen 7B profiles requires a CUDA-compatible NVIDIA GPU with sufficient memory.
2. **Select the source revision.** In the first configuration cell, set `REPO_REF` to the commit SHA you want to reproduce (for example, replace the default `draft` value with the SHA of the published version). You can also configure it with the `REPRO_REPO_REF` environment variable. The notebook records the resolved commit in `environment_manifest.json`.
3. **Choose the run mode.** `RUN_EXPERIMENTS` is enabled by default and runs all checks and experiments in order. `REGENERATE_LLM_PROFILES` defaults to `0`, so the notebook uses the saved profiles and embeddings without regenerating them. To regenerate profiles, set this environment variable to `1`, `true`, or `yes` before starting the kernel; this requires a compatible NVIDIA GPU. You can change `RUN_EXPERIMENTS` in the first configuration cell or set its environment variable before starting the kernel.
4. **Run all cells in order.** Open the notebook in Kaggle/Colab or Jupyter/VS Code and run every cell from top to bottom. The notebook detects the environment, finds an existing repository or clones it, installs/checks dependencies, downloads Git LFS data, fixes random seeds, checks the input files, and runs the pipeline. On a local machine, if the notebook is outside the repository directory, set `REPRO_REPO_PATH` to the checked-out repository directory.
5. **Review the results.** Each run saves an environment manifest and executed notebooks under `results/reproduction_runs/<run_id>/`. The runtime depends on the hardware; a full pipeline run may take several hours.

The notebook makes the workflow easier to reproduce across platforms, but successful execution cannot be guaranteed on every machine. Results depend on network access, GitHub/Git LFS permissions, compatible package versions, available memory, and hardware. For comparable results, use the same commit SHA, data, and experiment settings; the manifest records environment versions to help identify differences.

## 7. Reproducing the paper

| Paper section/table | Notebook or script | Output | Expected result / check |
|---|---|---|---|
| Table 1: splits | `src/data/02-data-splits.ipynb` | `data/splits/cold_scenario/` | 100,000 sampled users; 753 cold items (251 validation, 502 test) |
| Section 3.1: profiles and embeddings | `llm/course_profile.py` | `data/course_llm/2_profiles_full_and_shards/`, `data/course_llm/1_embeddings/` | 3,769 courses; 384-dimensional SBERT vectors |
| Section 3.2: masked prerequisites | `python llm/course_profile.py --masked_prereq_eval` | `data/course_llm/3_quality_evaluation/masked_prereq_200_eval_report.json` | 200-course evaluation report |
| Section 4.2: leakage sanity check | `src/models/lightgcn_leakage.ipynb` | Notebook output | Warm/cold temporal split check completes without leakage assertions |
| Section 4.4: LightGCN backbone | `src/models/cold_scenario/lightgcn_backbone_cold.ipynb` | `data/embeddings/01_lightgcn_backbone_cold/g_u.npy`, `g_i.npy` | MD5 prefixes `7a170558` and `d17ad9df` |
| Table 2: random and ID LightGCN | `src/eval/evaluate_test_random.ipynb`, `src/models/cold_scenario/lightgcn_id_cold.ipynb` | Notebook output; `results/random_ndcg_analytic.txt` | Saved empirical random means: 0.015705 on 502 cold candidates and 0.001983 over the full catalog. Analytic expectations: 0.01587648 and 0.00211462. ID LightGCN reference: 0.0165. |
| Content-KNN | `src/models/cold_scenario/content-knn-v2.ipynb` | `scores/Content-KNN/` | Fused/raw mean NDCG@20 difference about +0.0001; no significant RQ1 difference |
| DropoutNet | `src/models/cold_scenario/dropoutnet-v2.ipynb`, `src/models/cold_scenario/dropoutnet-v2-p-rq3.ipynb`, `src/models/cold_scenario/dropoutnet-llm-3d.ipynb` | `scores/DropoutNet/` | RQ1 fused/raw Δ=+0.0103; RQ2 new-course Δ=+0.0147; RQ3 fused-profile 5D−3D Δ=+0.0044 |
| ALDI | `src/models/coldrec-adapter.ipynb`, `src/models/run_aldi.py` | `scores/ALDI/` | RQ1 fused/raw Δ=−0.0018; not significant in the saved final comparison |
| SemProj | `src/models/semproj_experiments.ipynb`, `src/models/semproj.py` | `scores/SemProj/` | RQ1 fused/raw Δ=+0.0142; RQ2 new-course Δ=+0.0212; RQ3 fused-profile 5D−3D Δ=+0.0122 |
| Section 6.5: USIM | `src/models/coldrec-adapter-usim-raw.ipynb` | Notebook output and `scores/USIM/` when rerun | Check backbone loading, identity mapping, and cold-validation score before running test evaluation |
| Tables 2–4 and RQ verdicts | `results/final-stats.ipynb` | `results/final_stats.csv`, `results/verdicts.json` | Saved RQ1–RQ3 verdicts and paired statistical tests |
| Supplementary RQ4: inference latency | `src/models/semproj_experiments.ipynb`, `src/models/cold_scenario/content-knn-v2.ipynb` | Notebook output; SemProj writes `results_semproj/prq4_latency_gpu.csv` when rerun | Content-KNN measured 0.0024 ms/user for raw and LLM profiles, and 0.0035 ms/user for fused profiles; saved environment does not identify the GPU model |
| Sections 6.3/6.5: appendix checks | `src/eval/random_ndcg_analytic.py`, `src/eval/time_consistent_subset.py` | `results/random_ndcg_analytic.txt`, `results/time_consistent_subset.txt` | Analytic random expectations 0.01587648 (cold-only) and 0.00211462 (full catalog); time-consistent subset retains 1,402/5,531 test interactions (25.35%) |
| Appendix B.3: cold-item rule | `src/eval/check_cold_items.py` | Console report | Exact match compares the fixed JSON union against the latest 20% of courses |

The saved final comparison reports an RQ2 improvement concentrated on new courses, and a positive RQ3 five-dimension effect for DropoutNet and SemProj. Absolute means and confidence intervals are in `results/final_stats_tables_with_notes.csv`; `results/final_stats.csv` contains the paired-test details.

### External implementations

The notebooks pin the upstream repositories to the commits below:

- [ColdRec](https://github.com/YuanchenBei/ColdRec), MIT-licensed upstream: `1108192665aca2cb94b3aeb5bfed29850874eb1c`.
- [ALDI](https://github.com/zfnWong/ALDI): `87f9fe369427402bd989728c4d27b543b2d19cd4`. No license file was visible in the upstream repository listing reviewed; inspect upstream terms before redistributing its code.

`src/models/coldrec_fix.py` applies the local USIM compatibility changes: removes the stale SimCSR import, preserves original user/item IDs (including cold items missing from training CSVs), evaluates USIM on cold validation, and writes the LightGCN backbone and corrected `info_dict.pkl` expected by ColdRec.

## 8. Citation and licensing

The paper title is available in the project materials, but author, venue, year, and DOI metadata were not present in the supplied repository snapshot. Cite the paper using its final published metadata when available.

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

This repository snapshot has no root `LICENSE` file, so it does not declare an overall license for its code. MOOCCubeX upstream is marked GPL-3.0; ColdRec is marked MIT; check the respective upstream notices and the data-source terms when redistributing those materials.
