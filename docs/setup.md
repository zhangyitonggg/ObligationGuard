# Execution setup

Local data validation and unit tests use Python 3.11 or newer and the standard library. Full training and SusVibes execution use a Linux CUDA environment. Run commands from the repository root.

## Training and inference environments

Use a dedicated training environment for `requirements-train.txt` and a separate inference environment for `requirements-inference.txt`. The two files pin different PyTorch and Transformers dependency stacks through their respective packages. The package installation scripts do not train or invoke models.

```bash
export OBLIGATIONGUARD_CACHE_ROOT=/path/to/cache
export OBLIGATIONGUARD_RUN_ROOT=/path/to/runs
export OBLIGATIONGUARD_INFERENCE_ENV=/path/to/inference-env
bash scripts/setup_train.sh
python -m pip install --cache-dir "$OBLIGATIONGUARD_CACHE_ROOT/pip" -r requirements-agent.txt
bash scripts/setup_inference.sh
export OBLIGATIONGUARD_INFERENCE_PYTHON="$OBLIGATIONGUARD_INFERENCE_ENV/bin/python"
export HF_HOME="$OBLIGATIONGUARD_CACHE_ROOT/huggingface"
export TORCH_HOME="$OBLIGATIONGUARD_CACHE_ROOT/torch"
```

Existing `HF_HOME` and `TORCH_HOME` values take precedence in the execution scripts. Managed servers also inherit those values. `server.cuda_visible_devices` can select a separate inference GPU when the actor and a local guard must coexist during RQ4.

## Model interfaces

`configs/models.json` contains the model registry; `configs/model_sources.json` records official model references. Set the required credentials in environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `MOONSHOT_API_KEY`, `ZAI_API_KEY`, and `DASHSCOPE_API_KEY`. Set `QWEN_API_BASE_URL` to the Model Studio endpoint for the account's region. Authenticated model downloads use `HF_TOKEN` when required by the model provider.

Copy `configs/runtime.toml` to a file ending in `.local.toml` for machine-specific overrides. These files and `.env` are excluded from version control. Model credentials are read from environment variables and are not stored in configuration files.

## SusVibes

Prepare the official harness at the revision in `configs/external_sources.json`:

```bash
git clone https://github.com/LeiLiLab/susvibes /path/to/susvibes
git -C /path/to/susvibes checkout 520f16f9b3f39b06c6013bc0eb57a38280b4deee
export OBLIGATIONGUARD_SUSVIBES_CHECKOUT=/path/to/susvibes
python -m pip install --cache-dir "$OBLIGATIONGUARD_CACHE_ROOT/pip" -e "$OBLIGATIONGUARD_SUSVIBES_CHECKOUT"
python scripts/fetch_susvibes.py --output data/susvibes.jsonl
```

The adapter loads `config/default.yaml` from the pinned Mini-SWE-Agent package in `requirements-agent.txt`. Docker must be available to the Linux environment. The default snapshot mode uses Docker checkpoint/restore support and CRIU to preserve process state as well as files. `idle_container` mode is available only when the snapshot contains no persistent process other than the container's sleep process; the adapter checks this before capturing it.

## Inputs and checks

The canonical ObligationBench records are included at `data/obligationbench.jsonl`; see [the data format](data_format.md). Set `evaluation_models.ObligationGuard.model` to the selected trained checkpoint, or place the checkpoint at `$OBLIGATIONGUARD_RUN_ROOT/checkpoints/best`. Training can be invoked separately with the `train` command. The without-templates ablation includes 40,000 training and 2,000 validation examples under `data/without_templates/`, configured through `paths.direct_train` and `paths.direct_validation`. Its split follows the same trajectory-level random splitting and shuffling procedure and seed as the main training data. The retained source is configured as `paths.without_templates`; if the two prepared split settings are omitted, an explicitly invoked RQ3 command prepares the split in the run directory.

```bash
python -m obligationguard validate --input data/obligationbench.jsonl --benchmark
python -m obligationguard check --runtime configs/runtime.toml
python -m unittest discover -s tests -v
```

The check command does not invoke models. A return code of 2 indicates missing requirements or invalid data. After all requirements are present, `scripts/run_all.sh` evaluates the 120 positive instances and exits. It uses an existing checkpoint. The negative instances remain in the repository as data. Windows users can use `scripts/run_all.bat`, `scripts/check.bat`, and `scripts/test.bat`; CUDA training and container experiments require the configured Linux environment.

The [vLLM model registry](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/models/registry.py) and [tokenization protocol](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/entrypoints/serve/tokenize/protocol.py) define the managed server's supported architectures and context-counting request.

The Responses adapter uses the official [input-token counting interface](https://developers.openai.com/api/docs/guides/token-counting). The [reasoning-model documentation](https://developers.openai.com/api/docs/guides/reasoning) describes the `incomplete` status when generation reaches its token limit; the adapter retains that response and records its completion error.
