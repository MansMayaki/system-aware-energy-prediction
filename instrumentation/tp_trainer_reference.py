# energy_bert/distributed/tp_trainer.py

import os
import math
import json
import time
import csv
import datetime as dt
from contextlib import nullcontext
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

from transformers import (
    DataCollatorWithPadding,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
)

from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)

from energy_bert.data.tasks import GLUE_TASKS, QA_TASKS
from energy_bert.data.loaders import build_tokenizer, load_text_classification, load_qa
from energy_bert.data.collators import TokenCounter, CountingCollator
from energy_bert.models.custom_bert import (
    build_custom_bert_for_task,
    approx_flops_per_token,
    approx_bytes_proxy,
    approx_params_BERT,
)
from energy_bert.callbacks.token_accounting import TokenAccountingCallback
from energy_bert.energy.nvml_sampler import NVMLPowerSampler
from energy_bert.energy.codecarbon_tracker import EpochCodeCarbon


# ----------------- Loss recording -----------------


class LossRecorder:
    def __init__(self):
        self.rows = []  # list[dict]

    def log(self, *, epoch: int, step: int, global_step: int, loss: float, lr: float):
        self.rows.append(
            {
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "epoch": epoch,
                "step_in_epoch": step,
                "global_step": global_step,
                "loss": float(loss),
                "lr": float(lr),
            }
        )

    def save_csv(self, path: str):
        if not self.rows:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        keys = ["timestamp", "epoch", "step_in_epoch", "global_step", "loss", "lr"]
        newfile = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            if newfile:
                w.writeheader()
            for r in self.rows:
                w.writerow({k: r.get(k) for k in keys})
        self.rows.clear()


# ----------------- TP plan -----------------


def make_tp_plan_for_hf_bert(model: nn.Module) -> Dict[str, Any]:
    """
    Safer, minimal TP plan:
      - MLP intermediate.dense: column-wise
      - MLP output.dense (non-attn): row-wise

    This avoids touching attention projections, which keeps life simpler.
    """
    plan: Dict[str, Any] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "intermediate.dense" in name:
            plan[name] = ColwiseParallel()
        elif "output.dense" in name and "attention" not in name:
            plan[name] = RowwiseParallel()
    return plan


# ----------------- Train / eval loops -----------------


def train_one_epoch_tp(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    fp16: bool,
    bf16: bool,
    grad_accum: int,
    rank: int,
    token_cb: TokenAccountingCallback,
    epoch_idx: int,
    loss_recorder: Optional[LossRecorder],
    global_step0: int,
    base_lr: float,
    max_train_steps: int | None = None,

) -> int:
    model.train()
    step_in_epoch = 0
    global_step = global_step0

    use_amp = device.type == "cuda" and (fp16 or bf16)
    autocast_dtype = torch.float16 if fp16 else torch.bfloat16

    for batch in dataloader:
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

        ctx = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if use_amp
            else nullcontext()
        )

        with ctx:
            out = model(**batch)
            true_loss = out.loss
            loss = true_loss / grad_accum

        loss.backward()

        # current LR (for logging only; schedule applied outside)
        lr_now = optimizer.param_groups[0].get("lr", base_lr)

        if ((step_in_epoch + 1) % grad_accum) == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        # token accounting
        token_cb.on_train_batch_end(global_step, batch)

        if rank == 0 and loss_recorder is not None:
            loss_recorder.log(
                epoch=epoch_idx,
                step=step_in_epoch,
                global_step=global_step,
                loss=true_loss.detach().item(),
                lr=lr_now,
            )

        step_in_epoch += 1
        if max_train_steps is not None and global_step >= max_train_steps:
            break
    return global_step


def evaluate_tp(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            out = model(**batch)
            total += float(out.loss.item())
            n += 1
    return total / max(n, 1)


# ----------------- Main TP trainer -----------------


def run_tp_training(args):
    # ---- init dist ----
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo"
        )

    world = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world < 2:
        if rank == 0:
            print("[FATAL] Tensor Parallel training requires at least 2 processes (world_size >= 2).")
        raise SystemExit(1)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        device_type = "cuda"
    else:
        device = torch.device("cpu")
        device_type = "cpu"

    # For TP, all ranks must see the SAME batches → same seed
    torch.manual_seed(args.seed)

    if rank == 0:
        print(
            f"[TP] world_size={world}, rank={rank}, local_rank={local_rank}, device={device}"
        )

    # ---- data ----
    tokenizer = build_tokenizer("bert-base-uncased")
    is_qa = args.task in QA_TASKS

    if is_qa:
        datasets, base_collator = load_qa(args.task, tokenizer, args.max_length)
        train_ds = datasets["train"]
        eval_ds = None if args.no_eval else datasets.get("validation")
    else:
        datasets, num_labels = load_text_classification(
            args.task, tokenizer, args.max_length
        )
        train_ds = datasets["train"]
        eval_split = "validation_matched" if args.task == "mnli" else "validation"
        eval_ds = None if args.no_eval else datasets.get(eval_split)
        base_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    token_counter = TokenCounter()
    data_collator = CountingCollator(base_collator, token_counter)
    token_cb = TokenAccountingCallback(log_steps=args.log_steps)

    # No DistributedSampler → each rank sees same order (thanks to same seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        pin_memory=True,
    )
    eval_loader = (
        None
        if eval_ds is None
        else DataLoader(
            eval_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=data_collator,
            pin_memory=True,
        )
    )

    # ---- model ----
    if args.custom_bert:
        ff = args.ff if args.ff is not None else 4 * args.hidden
        num_labels_task = (
            2
            if is_qa
            else GLUE_TASKS.get(args.task, {"num_labels": 2})["num_labels"]
        )
        model, cfg = build_custom_bert_for_task(
            args.task, args.layers, args.hidden, args.heads, ff, num_labels_task
        )
        hidden_size = cfg.hidden_size
        num_layers = cfg.num_hidden_layers
        ff_size = cfg.intermediate_size
        params_calc = approx_params_BERT(num_layers, hidden_size, ff_size)
        model_name_for_row = f"CustomBERT(L={args.layers},d={args.hidden},h={args.heads},ff={ff})"
    else:
        cfg = AutoConfig.from_pretrained(args.model)
        hidden_size = getattr(cfg, "hidden_size", 768)
        num_layers = getattr(cfg, "num_hidden_layers", 12)
        ff_size = getattr(cfg, "intermediate_size", 3072)
        params_calc = approx_params_BERT(num_layers, hidden_size, ff_size)

        model_cls = (
            AutoModelForQuestionAnswering
            if is_qa
            else AutoModelForSequenceClassification
        )
        num_labels_task = (
            2
            if is_qa
            else GLUE_TASKS.get(args.task, {"num_labels": 2})["num_labels"]
        )
        model = model_cls.from_pretrained(args.model, num_labels=num_labels_task)
        model_name_for_row = args.model

    model.to(device)

    # ---- Tensor Parallel sharding ----
    device_mesh = init_device_mesh(device_type=device_type, mesh_shape=(world,))
    tp_plan = make_tp_plan_for_hf_bert(model)
    model = parallelize_module(model, device_mesh=device_mesh, parallelize_plan=tp_plan)

    if rank == 0:
        print(f"[TP] Applied TP plan: {len(tp_plan)} Linear layers sharded.")

    # ---- optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        foreach=False,
    )
    base_lr = args.lr

    # ---- output dirs ----
    run_id = args.run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"{args.task}_TP_{run_id}")
    """
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    """
    loss_recorder = None
    """
    loss_recorder = LossRecorder() if rank == 0 else None
    train_losses_csv = os.path.join(out_dir, "train_losses.csv")
    eval_losses_csv = os.path.join(out_dir, "eval_losses.csv")
    """
    summary_csv = os.path.join(args.output_dir, "bert_ft_runs_tp.csv")

    # ---- warmup schedule ----
    total_steps = math.ceil(len(train_loader) / max(1, args.grad_accum)) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    def lr_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return 1.0

    global_step = 0
    last_val_loss: Optional[float] = None

    # ---- epochs ----
    for epoch in range(args.epochs):
        if rank == 0:
            print(f"[TP] Running epoch {epoch} be patient")

        # per-epoch trackers
        cc = (
            EpochCodeCarbon(
                measure_power_secs=0.1,
                tracking_mode="process",
                log_level="error",
                save_to_file=False,
                output_dir=args.output_dir,
                project_name="bert_energy_tp",
            )
            if (args.enable_codecarbon and rank == 0)
            else None
        )
        nvml_sampler = (
            NVMLPowerSampler(interval_sec=0.1)
            if (args.enable_nvml and rank == 0 and torch.cuda.is_available())
            else None
        )

        # reset token stats
        token_counter.tokens_seen = 0
        token_counter.max_seq = 0
        token_counter.steps = 0
        token_cb.on_train_begin()

        if rank == 0 and cc:
            cc.start()
        if rank == 0 and nvml_sampler:
            nvml_sampler.start()

        # LR at start of epoch (global_step-aware warmup)
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr * lr_lambda(global_step)

        t0 = time.time()

        global_step = train_one_epoch_tp(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            fp16=args.fp16,
            bf16=args.bf16,
            grad_accum=args.grad_accum,
            rank=rank,
            token_cb=token_cb,
            epoch_idx=epoch,
            loss_recorder=loss_recorder,
            global_step0=global_step,
            base_lr=base_lr,
        )

        t1 = time.time()
        duration_s = t1 - t0

        # flush per-epoch train losses
        """
        if rank == 0 and loss_recorder is not None:
            loss_recorder.save_csv(train_losses_csv)
        """
        # finalize energy
        energy_kwh_cc = cc.stop() if (rank == 0 and cc) else None

        energy_kwh_nvml = None
        if rank == 0 and nvml_sampler:
            nvml_sampler.stop()
            energy_kwh_nvml = nvml_sampler.energy_kwh()

        # eval
        if eval_loader is not None and not args.no_eval:
            val_loss = evaluate_tp(model, eval_loader, device)
            last_val_loss = float(val_loss)
            print(f"[TP] validation_loss={val_loss:.4f}")
            """
            if rank == 0:
                print(f"[TP] validation_loss={val_loss:.4f}")
                newfile = not os.path.exists(eval_losses_csv)
                with open(eval_losses_csv, "a", newline="") as f:
                    w = csv.DictWriter(
                        f, fieldnames=["timestamp", "epoch", "val_loss"]
                    )
                    if newfile:
                        w.writeheader()
                    w.writerow(
                        {
                            "timestamp": dt.datetime.now().isoformat(
                                timespec="seconds"
                            ),
                            "epoch": epoch,
                            "val_loss": float(val_loss),
                        }
                    )
            """

        # ----- per-epoch metrics -----
        if rank == 0:
            tokens_seen = int(token_counter.tokens_seen)
            seq_len_batch_max = int(token_counter.max_seq)

            # throughput stats
            tps_series = token_cb.tokens_per_sec_series
            if tps_series:
                t_tensor = torch.tensor(tps_series)
                tps_p50 = float(t_tensor.median().item())
                tps_p90 = float(t_tensor.quantile(0.9).item())
            else:
                tps_p50 = tps_p90 = 0.0

            flops_token = approx_flops_per_token(
                num_layers, hidden_size, seq_len_batch_max or args.max_length
            )
            C_proxy = float(tokens_seen) * float(flops_token)
            M_proxy = approx_bytes_proxy(
                args.batch_size,
                seq_len_batch_max or args.max_length,
                hidden_size,
                num_layers,
            )
            tokens_per_sec = (tokens_seen / duration_s) if duration_s > 0 else 0.0

            machine_p90_env = os.environ.get("ENERGY_MACHINE_TPS_P90", "")
            try:
                machine_tps_p90 = (
                    float(machine_p90_env)
                    if machine_p90_env
                    else (tps_p90 or max(tokens_per_sec, 1e-6))
                )
            except Exception:
                machine_tps_p90 = tps_p90 or max(tokens_per_sec, 1e-6)

            eta_h = (tokens_per_sec / machine_tps_p90) if machine_tps_p90 > 0 else 1.0

            energy_kwh = (
                float(energy_kwh_cc) * float(args.power_correction)
                if energy_kwh_cc is not None
                else None
            )
            cuda_name = (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "cpu"
            )
            machine = args.machine_name or cuda_name

            row = {
                "run_id": run_id,
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "machine": machine,
                "gpus": world,
                "epoch": epoch,
                "val_loss": last_val_loss,
                "model": f"{model_name_for_row}+TP",
                "task": args.task.upper(),
                "energy_kwh": round(energy_kwh, 6)
                if energy_kwh is not None
                else None,
                "energy_kwh_nvml": round(energy_kwh_nvml, 6)
                if energy_kwh_nvml is not None
                else None,
                "duration_s": round(duration_s, 3),
                "tokens_seen": tokens_seen,
                "seq_len_batch": seq_len_batch_max,
                "batch_size": args.batch_size,
                "layers": num_layers,
                "hidden_size": hidden_size,
                "heads": args.heads if args.custom_bert else None,
                "ff": (
                    args.ff
                    if (args.custom_bert and args.ff is not None)
                    else (4 * hidden_size)
                ),
                "params_calc": params_calc,
                "flops_per_token_proxy": flops_token,
                "C_proxy": C_proxy,
                "M_proxy": M_proxy,
                "tokens_per_sec": round(tokens_per_sec, 3),
                "tokens_per_sec_p50": round(tps_p50, 3),
                "tokens_per_sec_p90": round(tps_p90, 3),
                "eta_h_proxy": round(eta_h, 6),
                "max_length": args.max_length,
                "grad_accum": args.grad_accum,
                "fp16": args.fp16,
                "bf16": args.bf16,
                "lr": args.lr,
                "warmup_ratio": args.warmup_ratio,
                "weight_decay": args.weight_decay,
                "power_correction": args.power_correction,
                "tp_world_size": world,
            }

            os.makedirs(args.output_dir, exist_ok=True)
            file_exists = os.path.exists(summary_csv)
            with open(summary_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not file_exists:
                    w.writeheader()
                w.writerow(row)
            """
            with open(os.path.join(out_dir, "run_metrics.json"), "w") as f:
                json.dump(row, f, indent=2)
            """
            

            print("\n=== TP RUN SUMMARY ===")
            for k, v in row.items():
                print(f"{k}: {v}")
            print(f"\nSaved row to: {summary_csv}\nOutputs dir: {out_dir}")

    # clean up
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
