"""Fault-aware acoustic training loop with AMP, accumulation, resume, and early stopping."""

from __future__ import annotations

import logging
import platform
import signal
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tts_pipeline.config import Settings
from tts_pipeline.losses.acoustic import acoustic_loss
from tts_pipeline.training.checkpoint import load_checkpoint, save_checkpoint
from tts_pipeline.training.tracker import Tracker
from tts_pipeline.utils.runtime import git_revision, seed_everything

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_validation: float = float("inf")
    stale_epochs: int = 0


class AcousticTrainer:
    def __init__(
        self, model: torch.nn.Module, settings: Settings, tracker: Tracker, run_dir: Path
    ) -> None:
        self.model = model
        self.settings = settings
        self.tracker = tracker
        self.run_dir = run_dir
        self.device = torch.device(settings.runtime.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=settings.training.learning_rate,
            weight_decay=settings.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.999)
        self.scaler = torch.amp.GradScaler(  # type: ignore[attr-defined]
            "cuda", enabled=settings.training.mixed_precision and self.device.type == "cuda"
        )
        self.state = TrainState()
        self.stop_requested = False

    def startup_report(
        self, train_size: int, validation_size: int, speakers: int, vocabulary: int
    ) -> dict[str, Any]:
        return {
            "config": self.settings.model_dump(mode="json"),
            "git": git_revision(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "device": str(self.device),
            "device_name": torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda"
            else "CPU",
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "train_size": train_size,
            "validation_size": validation_size,
            "speakers": speakers,
            "vocabulary": vocabulary,
            "features": self.settings.audio.model_dump(),
            "seed": self.settings.runtime.seed,
        }

    def resume(self, checkpoint: Path) -> None:
        raw = load_checkpoint(checkpoint, self.model, self.optimizer, self.scheduler, self.scaler)
        self.state = TrainState(**raw)

    def fit(
        self,
        train_loader: Iterable[dict[str, torch.Tensor]],
        validation_loader: Iterable[dict[str, torch.Tensor]],
    ) -> TrainState:
        seed_everything(self.settings.runtime.seed, self.settings.runtime.deterministic)
        previous = signal.signal(signal.SIGINT, lambda *_: setattr(self, "stop_requested", True))
        try:
            for epoch in range(self.state.epoch, self.settings.training.epochs):
                self.state.epoch = epoch
                train_metric = self._epoch(train_loader, training=True)
                validation_metric = self._epoch(validation_loader, training=False)
                self.tracker.log_metrics(
                    {"train_loss": train_metric, "validation_loss": validation_metric},
                    self.state.global_step,
                )
                self.scheduler.step()
                if validation_metric < self.state.best_validation:
                    self.state.best_validation = validation_metric
                    self.state.stale_epochs = 0
                    self._checkpoint("best.pt")
                else:
                    self.state.stale_epochs += 1
                self._checkpoint("latest.pt")
                if (
                    self.stop_requested
                    or self.state.stale_epochs >= self.settings.training.patience
                ):
                    break
        finally:
            signal.signal(signal.SIGINT, previous)
            self._checkpoint("latest.pt")
        return self.state

    def _epoch(self, loader: Iterable[dict[str, torch.Tensor]], training: bool) -> float:
        self.model.train(training)
        total, batches = 0.0, 0
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader):
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with (
                torch.set_grad_enabled(training),
                torch.amp.autocast(  # type: ignore[attr-defined]
                    self.device.type, enabled=self.scaler.is_enabled()
                ),
            ):
                output = self.model(
                    batch["tokens"],
                    batch["token_lengths"],
                    duration_targets=batch["durations"],
                    pitch_targets=batch["pitch"],
                    energy_targets=batch["energy"],
                )
                losses = acoustic_loss(
                    output, batch["mel"], batch["durations"], batch["pitch"], batch["energy"]
                )
                loss = losses["total"] / self.settings.training.gradient_accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {self.state.global_step}")
            if training:
                self.scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
                if (batch_index + 1) % self.settings.training.gradient_accumulation == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.settings.training.gradient_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.state.global_step += 1
                    if self.state.global_step % self.settings.training.checkpoint_every == 0:
                        self._checkpoint(f"step-{self.state.global_step}.pt")
            total += float(losses["total"].detach())
            batches += 1
            if self.stop_requested:
                break
        if not batches:
            raise ValueError("data loader produced no batches")
        return total / batches

    def _checkpoint(self, name: str) -> None:
        path = save_checkpoint(
            self.run_dir / name,
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            **self.state.__dict__,
            config_fingerprint=self.settings.fingerprint(),
        )
        self.tracker.log_artifact(path)
