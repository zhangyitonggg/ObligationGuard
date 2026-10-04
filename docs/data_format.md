# Data format

A canonical instance has the following fields:

| Field | Type | Meaning |
|---|---|---|
| `instance_id` | string | Opaque identifier used only for local record association |
| `task` | string | User task |
| `trajectory` | array of objects | Complete recorded execution |
| `obligations` | array of objects | Ground-truth actions remaining unfulfilled |
| `scenario_id` | string | Base scenario identifier for training examples |
| `source_scenario_ids` | array of strings | Source scenarios used in a composed example |
| `pair_id` | string | Optional positive/negative task-pair identifier |
| `task_domain` | string | `issue_resolution`, `feature_development`, or `terminal_operation` |
| `metadata` | object | Source information excluded from model inputs |

An obligation uses `id`, `required_safety_action`, `safety_consequence`, and `evidence`. The optional `creation_step` identifies the earliest action/observation step at which the obligation arises. The optional `safety_category` is one of the categories in Section 3.2. RQ2 annotates missing creation steps separately; ordinary obligation identification does not require either field.

An empty `obligations` array denotes a negative instance. Model inputs contain the user task and complete trajectory, with the fixed JSON placeholder `task_id: "task"`. The same placeholder is used in training targets and validated model outputs. Dataset identifiers, ground-truth annotations, and source metadata are excluded from model inputs. Predictions and scores are associated with each sample through the local `instance_id`, which is not sent to the model. Imported identifiers use the same opaque format for positive and negative instances.

The collection importer maps external action and consequence field names to these fields. It retains the target and source identifiers in metadata, and removes private contact addresses and user-directory identifiers from both keys and values during import. The supplied source files are not modified.

The benchmark importer reads `positive/` and `negative/` sample directories, each containing `guard_input.json`, `ground_truth.json`, and `provenance.json`. It retains complete message histories and the supplied obligation labels, including evidence message indices. The first user message supplies the task. Source benchmark names determine the task domain. The one-click entry point selects instances with nonempty obligations before prediction and semantic judging.

For creation-distance analysis, recorded tool calls are grouped with their corresponding observations into sequential action steps. Original message indices and contextual messages are retained. The distance is measured in actions, following Section 6.2.
