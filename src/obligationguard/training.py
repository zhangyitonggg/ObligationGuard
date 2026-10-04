from __future__ import annotations

import json
import os
from pathlib import Path

from .backends import completion_result, make_backend
from .datasets import load_instances, validate_training_split
from .errors import ConfigurationError, DataError, ModelOutputError
from .evaluation import evaluate
from .io import canonical_json, write_json, write_jsonl
from .prompting import identification
from .schema import Instance


def encode_target_only(instance: Instance, tokenizer, max_length: int, chat_template_kwargs: dict) -> dict:
    prompt = identification(instance)
    target_json = canonical_json(instance.target())
    completed = tokenizer.apply_chat_template([{"role": "user", "content": prompt}, {"role": "assistant", "content": target_json}], tokenize=False, add_generation_prompt=False, **chat_template_kwargs)
    target_start = completed.rfind(target_json)
    if target_start < 0:
        raise ConfigurationError("official chat template changed the target JSON")
    prefix = completed[:target_start]
    if not isinstance(tokenizer.eos_token, str):
        raise ConfigurationError("tokenizer must define its assistant end token")
    target = target_json + tokenizer.eos_token
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    ids = tokenizer.encode(prefix + target, add_special_tokens=False)
    if ids[:len(prefix_ids)] != prefix_ids:
        raise DataError("chat template does not preserve the input/target token boundary")
    if len(ids) > max_length:
        raise DataError(f"{instance.instance_id}: complete training sequence has {len(ids)} tokens; limit is {max_length}")
    if len(ids) <= len(prefix_ids):
        raise DataError("training sequence contains no target tokens")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(prefix_ids) + ids[len(prefix_ids):]}


def collate_target_only(features: list[dict], pad_token_id: int) -> dict:
    import torch
    length = max(len(x["input_ids"]) for x in features)
    return {
        key: torch.tensor([x[key] + [padding] * (length - len(x[key])) for x in features], dtype=torch.long)
        for key, padding in (("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100))
    }


def train(paper: dict, runtime: dict) -> None:
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    settings = paper["training"]
    implementation = runtime["training_runtime"]
    paths = runtime["paths"]
    required = {"seed", "adam_beta1", "adam_beta2", "adam_epsilon", "max_output_tokens", "chat_template_kwargs"}
    if "seed" not in implementation:
        manifest_path = Path(paths["train"]).parent / "manifest.json"
        if manifest_path.is_file():
            implementation = {**implementation, "seed": json.loads(manifest_path.read_text(encoding="utf-8"))["seed"]}
    if not required <= implementation.keys():
        raise ConfigurationError(f"training_runtime requires {sorted(required - implementation.keys())}")
    if "Qwen3" in settings["model"] and implementation["chat_template_kwargs"].get("enable_thinking") is not True:
        raise ConfigurationError("Qwen3 training requires enable_thinking = true")
    effective_batch = settings["world_size"] * settings["per_device_batch_size"] * settings["gradient_accumulation_steps"]
    if effective_batch != settings["effective_batch_size"]:
        raise ConfigurationError("effective batch size does not match the distributed training configuration")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != settings["world_size"]:
        raise ConfigurationError(f"launch training with {settings['world_size']} processes")
    if not torch.cuda.is_available():
        raise ConfigurationError("full-parameter training requires CUDA")
    train_instances = load_instances(paths["train"])
    validation = load_instances(paths["validation"])
    validate_training_split(train_instances, validation, settings["train_examples"], settings["validation_examples"])
    set_seed(implementation["seed"])
    tokenizer = AutoTokenizer.from_pretrained(settings["model"], revision=implementation.get("model_revision"))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    template_kwargs = implementation["chat_template_kwargs"]
    limit = settings["max_sequence_length"]

    class TargetDataset(Dataset):
        def __init__(self, instances):
            self.instances = instances

        def __len__(self):
            return len(self.instances)

        def __getitem__(self, index):
            return encode_target_only(self.instances[index], tokenizer, limit, template_kwargs)

    train_data, validation_data = TargetDataset(train_instances), TargetDataset(validation)
    for instances in (train_instances, validation):
        for instance in instances:
            encode_target_only(instance, tokenizer, limit, template_kwargs)
    for instance in validation:
        encoded = tokenizer.apply_chat_template([{"role": "user", "content": identification(instance)}], tokenize=True, add_generation_prompt=True, **template_kwargs)
        if len(encoded) + implementation["max_output_tokens"] > limit:
            raise DataError(f"{instance.instance_id}: complete validation input exceeds context budget")
    arguments = TrainingArguments(
        output_dir=paths["checkpoints"],
        num_train_epochs=settings["epochs"],
        per_device_train_batch_size=settings["per_device_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        learning_rate=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
        warmup_ratio=settings["warmup_ratio"],
        lr_scheduler_type=settings["lr_scheduler_type"],
        adam_beta1=implementation["adam_beta1"],
        adam_beta2=implementation["adam_beta2"],
        adam_epsilon=implementation["adam_epsilon"],
        optim="adamw_torch",
        bf16=settings["bf16"],
        gradient_checkpointing=settings["gradient_checkpointing"],
        max_grad_norm=settings["max_grad_norm"],
        deepspeed=settings["deepspeed"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
        greater_is_better=True,
        seed=implementation["seed"],
        data_seed=implementation["seed"],
        report_to=[],
        remove_unused_columns=False,
    )

    class ObligationTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs["labels"]
            outputs = model(**{key: value for key, value in inputs.items() if key != "labels"})
            shifted_labels = labels[..., 1:].contiguous()
            logits = outputs.logits[..., :-1, :].contiguous()
            token_losses = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.shape[-1]), shifted_labels.view(-1), ignore_index=-100, reduction="none")
            loss = token_losses.view(shifted_labels.shape).sum(dim=-1).mean()
            return (loss, outputs) if return_outputs else loss

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            import deepspeed
            from accelerate.utils import gather_object
            import torch.distributed as distributed
            self.model.eval()
            unwrapped = self.accelerator.unwrap_model(self.model_wrapped)
            shard = validation[self.accelerator.process_index::self.accelerator.num_processes]
            local_predictions = []
            with deepspeed.zero.GatheredParameters(list(unwrapped.parameters()), modifier_rank=None):
                for instance in shard:
                    prompt = identification(instance)
                    encoded = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, return_tensors="pt", **template_kwargs).to(self.accelerator.device)
                    if encoded.shape[-1] + implementation["max_output_tokens"] > limit:
                        raise DataError(f"{instance.instance_id}: complete validation input exceeds context budget")
                    with torch.inference_mode():
                        generated = unwrapped.generate(encoded, attention_mask=torch.ones_like(encoded), do_sample=False, max_new_tokens=implementation["max_output_tokens"], pad_token_id=tokenizer.pad_token_id, synced_gpus=True)
                    output_tokens = generated[0, encoded.shape[-1]:]
                    text = tokenizer.decode(output_tokens, skip_special_tokens=True)
                    eos = unwrapped.generation_config.eos_token_id
                    eos_ids = eos if isinstance(eos, list) else [eos]
                    error = "validation output reached its token limit" if len(output_tokens) == implementation["max_output_tokens"] and output_tokens[-1].item() not in eos_ids else None
                    completion = completion_result(text, {"output_token_ids": output_tokens.tolist()}, {}, error)
                    local_predictions.append({"instance_id": instance.instance_id, "text": completion.text, "raw_response": completion.raw, "completion_error": completion.error})
            predictions = gather_object(local_predictions)
            directory = Path(paths["checkpoints"]) / f"validation-step-{self.state.global_step}"
            payload = [None]
            if self.is_world_process_zero():
                try:
                    write_jsonl(directory / "predictions.jsonl", predictions)
                    judge = make_backend(runtime["models"]["judge"])
                    result = evaluate(validation, directory / "predictions.jsonl", judge, directory / "judgments.jsonl", directory / "metrics.json")
                    payload[0] = {"metrics": result["metrics"]}
                except Exception as exc:
                    payload[0] = {"error": f"{type(exc).__name__}: {exc}"}
            distributed.broadcast_object_list(payload, src=0)
            if "error" in payload[0]:
                raise ModelOutputError(payload[0]["error"])
            metrics = {f"{metric_key_prefix}_{key}": value for key, value in payload[0]["metrics"].items() if key != "counts" and value is not None}
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics

    trainer = ObligationTrainer(
        model_init=lambda: AutoModelForCausalLM.from_pretrained(settings["model"], revision=implementation.get("model_revision"), torch_dtype=torch.bfloat16, attn_implementation=settings["attention_implementation"]),
        args=arguments,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=lambda features: collate_target_only(features, tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    if any(not parameter.requires_grad for parameter in trainer.model.parameters()):
        raise ConfigurationError("all backbone parameters must remain trainable")
    trainer.model.config.use_cache = False
    trainer.train(resume_from_checkpoint=implementation.get("resume_from_checkpoint"))
    trainer.save_model(str(Path(paths["checkpoints"]) / "best"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(Path(paths["checkpoints"]) / "best")
        write_json(Path(paths["checkpoints"]) / "training_settings.json", {"paper": settings, "runtime": implementation, "best_checkpoint": trainer.state.best_model_checkpoint, "best_validation_exact_match": trainer.state.best_metric})
