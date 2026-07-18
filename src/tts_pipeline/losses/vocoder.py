"""Least-squares adversarial and feature matching objectives."""

from __future__ import annotations

import torch

DiscriminatorOutput = list[tuple[torch.Tensor, list[torch.Tensor]]]


def discriminator_loss(real: DiscriminatorOutput, fake: DiscriminatorOutput) -> torch.Tensor:
    if not real:
        raise ValueError("discriminator output cannot be empty")
    total = real[0][0].new_zeros(())
    for real_item, fake_item in zip(real, fake, strict=True):
        total = total + torch.mean((1 - real_item[0]) ** 2) + torch.mean(fake_item[0] ** 2)
    return total


def generator_loss(fake: DiscriminatorOutput) -> torch.Tensor:
    if not fake:
        raise ValueError("discriminator output cannot be empty")
    total = fake[0][0].new_zeros(())
    for score, _ in fake:
        total = total + torch.mean((1 - score) ** 2)
    return total


def feature_matching_loss(real: DiscriminatorOutput, fake: DiscriminatorOutput) -> torch.Tensor:
    if not real:
        raise ValueError("discriminator output cannot be empty")
    total = real[0][0].new_zeros(())
    for (_, real_features), (_, fake_features) in zip(real, fake, strict=True):
        for real_feature, fake_feature in zip(real_features, fake_features, strict=True):
            total = total + torch.mean(torch.abs(real_feature.detach() - fake_feature))
    return total
