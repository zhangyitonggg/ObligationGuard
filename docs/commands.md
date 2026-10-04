# Commands

| Command | Function |
|---|---|
| `prepare` | Import the trajectory collection, randomly split it, and shuffle both sets |
| `import-benchmark` | Import benchmark sample directories and delete private identifiers |
| `validate` | Validate a canonical dataset; `--benchmark` checks ObligationBench statistics |
| `check` | Check positive-evaluation requirements without model calls; `--scope pipeline` checks the full research workflow |
| `positive` | Evaluate positive ObligationBench instances and exit; used by the one-click scripts |
| `train` | Run full-parameter SFT with validation EM checkpoint selection |
| `predict` | Identify obligations once per instance |
| `evaluate` | Judge obligation pairs and compute one-to-one metrics |
| `jobs` | Run a supplied synthesis or annotation job list |
| `rq2` | Analyze creation distance and concurrent obligations |
| `experiment` | Explicitly run an individual RQ1, RQ2, RQ3, or RQ4 experiment |
| `pipeline` | Explicitly run training and the configured research stages |
| `audit` | Scan repository filenames, field names, text, and PDF metadata for private identifiers |

Run `python -m obligationguard <command> --help` for arguments.

Prompt jobs use `job_id` and an `inputs` object. Its keys are exactly the placeholders of the corresponding prompt. Trajectory-synthesis jobs also carry `scenario_id`, linking them to their scenario-planning output. Rejected generations are preserved as rejected job outputs and excluded from assembled training examples.

The backends are `local`, `responses`, `chat`, `managed_chat`, and `anthropic`. Set `model`, `max_output_tokens`, and interface-specific settings. `chat` uses the configured tokenizer or the interface's token-count endpoint. `managed_chat` starts a separate local vLLM server and uses its tokenizer endpoint. Semantic matching uses the configured GPT-5.6-Sol judge at temperature 0. The full context is checked before generation; overlength inputs are not silently truncated.

Build a source-and-data ZIP using `python scripts/package_release.py --output /path/to/release --data /path/to/prepared-data`. The packager validates the data, scans the release for private identifiers, records file checksums, and verifies archive integrity. `--identity-env` names an environment variable containing additional private identity terms; those terms are not written to the release. The archive contains relative paths and excludes Git history, credentials, local configurations, and generated caches.

To update checksums and package the current repository without creating another source copy, use `python scripts/package_release.py --output . --data data --in-place --archive /path/to/release.zip`. The ZIP's top-level directory uses the output folder's name.

Install `requirements-audit.txt` to inspect the supplementary PDF during an anonymity audit. The audit checks its visible vector text, document metadata, and annotations. Uninspected binary content causes the audit to fail.

The README's paper figures were visually reviewed before inclusion. `docs/assets/sources.json` records their source figures and hashes; the audit verifies those hashes and scans image metadata. New or changed raster images require visual review before inclusion.
