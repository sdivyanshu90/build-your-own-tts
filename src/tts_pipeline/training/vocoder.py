"""Complete HiFi-GAN adversarial trainer with independently restorable state."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import torch
from torch.nn import functional as F

from tts_pipeline.audio import AudioProcessor
from tts_pipeline.config import Settings
from tts_pipeline.losses.vocoder import discriminator_loss, feature_matching_loss, generator_loss
from tts_pipeline.models.vocoder import (
    HiFiGANGenerator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
)
from tts_pipeline.training.checkpoint import sha256_file


class VocoderTrainer:
    def __init__(self, generator: HiFiGANGenerator, settings: Settings, run_dir: Path) -> None:
        self.generator = generator
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()
        self.settings = settings
        self.device = torch.device(settings.runtime.device)
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.generator.to(self.device)
        self.mpd.to(self.device)
        self.msd.to(self.device)
        self.generator_optimizer = torch.optim.AdamW(
            generator.parameters(), settings.training.learning_rate, betas=(0.8, 0.99)
        )
        discriminator_parameters = [*self.mpd.parameters(), *self.msd.parameters()]
        self.discriminator_optimizer = torch.optim.AdamW(
            discriminator_parameters, settings.training.learning_rate, betas=(0.8, 0.99)
        )
        self.generator_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.generator_optimizer, 0.999
        )
        self.discriminator_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.discriminator_optimizer, 0.999
        )
        self.scaler = torch.amp.GradScaler(  # type: ignore[attr-defined, unused-ignore]
            "cuda", enabled=settings.training.mixed_precision and self.device.type == "cuda"
        )
        self.processor = AudioProcessor(settings.audio)
        self.epoch = 0
        self.global_step = 0

    def fit(self, loader: Iterable[dict[str, torch.Tensor]]) -> None:
        for epoch in range(self.epoch, self.settings.training.epochs):
            self.epoch = epoch
            for batch in loader:
                mel = batch["mel"].to(self.device)
                real = batch["waveform"].to(self.device)
                with torch.amp.autocast(  # type: ignore[attr-defined, unused-ignore]
                    self.device.type, enabled=self.scaler.is_enabled()
                ):
                    fake = self.generator(mel)
                    length = min(real.shape[-1], fake.shape[-1])
                    real, fake = real[..., :length], fake[..., :length]
                    real_outputs = self.mpd(real) + self.msd(real)
                    fake_outputs = self.mpd(fake.detach()) + self.msd(fake.detach())
                    discriminator = discriminator_loss(real_outputs, fake_outputs)
                self.discriminator_optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(discriminator).backward()  # type: ignore[no-untyped-call]
                self.scaler.step(self.discriminator_optimizer)
                with torch.amp.autocast(  # type: ignore[attr-defined, unused-ignore]
                    self.device.type, enabled=self.scaler.is_enabled()
                ):
                    real_outputs = self.mpd(real) + self.msd(real)
                    fake_outputs = self.mpd(fake) + self.msd(fake)
                    adversarial = generator_loss(fake_outputs)
                    features = feature_matching_loss(real_outputs, fake_outputs)
                    generated_mel = self.processor.mel(fake.squeeze(1).float())
                    target_mel = self.processor.mel(real.squeeze(1).float())
                    frames = min(generated_mel.shape[-1], target_mel.shape[-1])
                    mel_loss = F.l1_loss(generated_mel[..., :frames], target_mel[..., :frames])
                    generator = adversarial + 2 * features + 45 * mel_loss
                self.generator_optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(generator).backward()  # type: ignore[no-untyped-call]
                self.scaler.step(self.generator_optimizer)
                self.scaler.update()
                self.global_step += 1
                if self.global_step % self.settings.training.checkpoint_every == 0:
                    self.save(self.run_dir / f"step-{self.global_step}.pt")
            self.generator_scheduler.step()
            self.discriminator_scheduler.step()
            self.save(self.run_dir / "latest.pt")

    def save(self, path: Path) -> None:
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "format_version": 1,
                "generator": self.generator.state_dict(),
                "mpd": self.mpd.state_dict(),
                "msd": self.msd.state_dict(),
                "generator_optimizer": self.generator_optimizer.state_dict(),
                "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
                "generator_scheduler": self.generator_scheduler.state_dict(),
                "discriminator_scheduler": self.discriminator_scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "epoch": self.epoch,
                "global_step": self.global_step,
            },
            temporary,
        )
        temporary.replace(path)
        path.with_suffix(".pt.json").write_text(
            json.dumps({"sha256": sha256_file(path), "format_version": 1}, indent=2) + "\n"
        )

    def resume(self, path: Path) -> None:
        manifest = json.loads(path.with_suffix(".pt.json").read_text())
        if sha256_file(path) != manifest["sha256"]:
            raise ValueError("vocoder checkpoint integrity failure")
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.generator.load_state_dict(state["generator"])
        self.mpd.load_state_dict(state["mpd"])
        self.msd.load_state_dict(state["msd"])
        self.generator_optimizer.load_state_dict(state["generator_optimizer"])
        self.discriminator_optimizer.load_state_dict(state["discriminator_optimizer"])
        self.generator_scheduler.load_state_dict(state["generator_scheduler"])
        self.discriminator_scheduler.load_state_dict(state["discriminator_scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.epoch, self.global_step = state["epoch"], state["global_step"]
