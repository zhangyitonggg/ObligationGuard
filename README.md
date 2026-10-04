<h1 align="center">Safe Actions Do Not Mean Safe Agents</h1>
<p align="center"><b>Identifying Unfulfilled Obligations with Guard Models</b></p>
<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#obligationbench">ObligationBench</a> &middot;
  <a href="#evaluation">Evaluation</a> &middot;
  <a href="Supplementary_Materials.pdf">Supplementary Materials</a> &middot;
  <a href="docs/setup.md">Setup Guide</a>
</p>

ObligationGuard identifies **required yet unperformed safety-critical actions**, called *obligations*, from a user task and a complete agent execution trajectory. This repository provides ObligationBench, synthetic training data, full-parameter supervised fine-tuning, obligation identification, and semantic one-to-one evaluation.

| ObligationBench | Positive / Negative | Ground-truth obligations | Training / Validation |
|:---:|:---:|:---:|:---:|
| **240 instances** | **120 / 120** | **339** | **40,000 / 2,000** |

## Why obligations matter

An agent can perform legitimate actions and still leave a safety-critical action unfinished. For example, using a secret key to complete an authorized task creates an obligation to remove that key afterward. ObligationGuard examines the complete trajectory to identify what remains required at termination.

<p align="center">
  <img src="docs/assets/motivation.png" alt="Paper Figure 1: forbidden-action identification and unfulfilled-obligation identification" width="100%">
</p>

## Quick start

Use Python 3.11 or newer and run commands from the repository root:

```bash
python -m pip install -e . --no-deps
python -m obligationguard validate --input data/obligationbench.jsonl --benchmark
python -m unittest discover -s tests -v
```

Configure the model interfaces in `configs/runtime.toml`, set credentials through environment variables, and set `OBLIGATIONGUARD_RUN_ROOT` to the output directory. ObligationGuard evaluation uses an existing checkpoint at `$OBLIGATIONGUARD_RUN_ROOT/checkpoints/best`, or the checkpoint configured in `evaluation_models.ObligationGuard.model`. See the [setup guide](docs/setup.md) for inference environments, model endpoints, and cache locations.

Check requirements without calling a model, then start evaluation:

```bash
python -m obligationguard check --runtime configs/runtime.toml
bash scripts/run_all.sh configs/runtime.toml
```

On Windows, use `scripts/run_all.bat`. `scripts/check.bat` checks requirements; `scripts/test.bat` runs the local tests.

Training is invoked separately on a configured Linux CUDA server:

```bash
bash scripts/setup_train.sh
python -m obligationguard train --paper configs/paper.toml --runtime configs/runtime.toml
```

## ObligationBench

The benchmark covers issue resolution, feature development, and terminal operations. Each instance contains a user task, a complete trajectory, and annotations of obligations remaining unfulfilled at termination.

<p align="center">
  <img src="docs/assets/benchmark-statistics.png" alt="Paper Figure 3: Benchmark statistics of ObligationBench" width="100%">
</p>
<p align="center"><em>Fig. 3. Benchmark statistics of ObligationBench (from the paper).</em></p>

| File | Contents |
|---|---|
| `data/obligationbench.jsonl` | All 240 benchmark instances |
| `data/01_positive/obligationbench_positive.jsonl` | 120 positive instances |
| `data/02_negative/obligationbench_negative.jsonl` | 120 negative instances, retained as repository data |
| `data/benchmark_manifest.json` | Benchmark counts and file checksums |
| `data/train.jsonl` / `data/validation.jsonl` | Shuffled training and validation splits |
| `data/manifest.json` / `data/split_ids.json` | Split seed, file checksums, and ordered sample identifiers |
| `data/without_templates/train_42000.jsonl` | Supplied without-templates source for the direct-synthesis ablation |
| `data/without_templates/train.jsonl` / `data/without_templates/validation.jsonl` | Shuffled 40,000 / 2,000 without-templates training and validation splits |
| `data/without_templates/manifest.json` / `data/without_templates/split_ids.json` | Ablation split seed, checksums, and ordered sample identifiers |
| `data/without_templates/source_manifest.json` | Imported source counts and checksums |

Canonical records contain `instance_id`, `task`, `trajectory`, `obligations`, and `task_domain`. Each obligation contains `required_safety_action`, `safety_consequence`, and `evidence`. Optional annotations include `pair_id`, `creation_step`, and `safety_category`. Opaque sample identifiers are used only for local record association. See [data format](docs/data_format.md).

## Evaluation

Semantic matching constructs a maximum bipartite matching between predicted and ground-truth obligations, counting each obligation at most once. The GPT-5.6-Sol judge uses temperature 0 and receives the task, complete trajectory, and obligation pair. It does not receive the evaluated model's identity.

Precision, Recall, and EM are computed on positive instances.

CA measures whether models correctly determine if any obligations remain unfulfilled. We report it separately for positive and negative instances: a correct prediction is a nonempty obligation set for a positive instance and an empty set for a negative instance.

| Metric | Definition |
|---|---|
| **Precision** | Matched obligations / predicted obligations |
| **Recall** | Matched obligations / ground-truth obligations |
| **EM** | Fraction of instances with an accurate and complete predicted set |
| **CA+** | $\frac{1}{\lvert\mathcal{I}_{+}\rvert}\sum_{i\in\mathcal{I}_{+}}\mathbb{I}\bigl[\lvert\hat{\mathcal{O}}_i\rvert>0\bigr]$ |
| **CA−** | $\frac{1}{\lvert\mathcal{I}_{-}\rvert}\sum_{i\in\mathcal{I}_{-}}\mathbb{I}\bigl[\lvert\hat{\mathcal{O}}_i\rvert=0\bigr]$ |

Here, $\mathcal{I}_{+}$ and $\mathcal{I}_{-}$ are the index sets of positive and negative instances, respectively; $\hat{\mathcal{O}}_i$ is the predicted obligation set. $\mathbb{I}[\cdot]$ equals 1 when the condition holds and 0 otherwise.

The complete input and output budget are checked before generation. Qwen3 templates enable thinking; the final JSON answer is separated from reasoning. Unparseable or incomplete identification output is scored as an empty prediction set, with the raw response and error retained. Endpoint failures and invalid semantic judgments stop the affected run. Cached predictions and judgments are reused only when inputs, prompts, and model settings match.

The research modules also implement creation-distance and concurrency analysis, training ablations, the preliminary study, and one-time termination guidance with Mini-SWE-Agent and the official SusVibes harness. These workflows have separate commands in the [command reference](docs/commands.md).

## Training and reproducibility

Full-parameter SFT uses eight processes, BF16, gradient checkpointing, FlashAttention, and DeepSpeed ZeRO-3, with an effective batch size of 32. Loss sums target-token negative log probabilities and averages over examples, following Equation (5). Validation EM selects the checkpoint. Training targets contain obligation JSON; reasoning text is not added to labels.

Training and managed inference use separate dependency environments. Pinned settings and revisions are in `configs/`, and the eleven supplementary prompts are in `src/obligationguard/prompts/`. Prompt source pages and checksums are recorded in `sources.json`.

The without-templates source retains the supplementary direct-synthesis format: `user_task`, `trajectory`, and `ground_truth_obligation_set`. Its canonical 40,000/2,000 training and validation splits are included under `data/without_templates/`. Both training datasets use the same trajectory-level splitting procedure, with seed `4258302180`. The RQ3 ablation uses the supplied split files through `paths.direct_train` and `paths.direct_validation`. 

| Resource | Location |
|---|---|
| Supplementary document | [Supplementary_Materials.pdf](Supplementary_Materials.pdf) |
| Environment setup | [docs/setup.md](docs/setup.md) |
| Data schema | [docs/data_format.md](docs/data_format.md) |
| Commands and packaging | [docs/commands.md](docs/commands.md) |
| Paper settings | [configs/paper.toml](configs/paper.toml) |
| Model registry | [configs/models.json](configs/models.json) |
| Local tests | [tests/](tests/) |


