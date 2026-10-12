import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import configs
from data import load_gray01, records_for_patients, require_gate, worker_count
from noise import apply_gaussian_filter, apply_noise, assert_noise_repeatable
from train_classifier import pick_device, set_seed


class ResidualDenoiser(nn.Module):
    def __init__(self, channels=configs.DENOISER_CHANNELS):
        super().__init__()
        layers = []
        in_channels = 1
        for index, out_channels in enumerate(channels):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if index < len(channels) - 1:
                layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, noisy):
        return torch.clamp(noisy + self.net(noisy), 0, 1)


class PairSet(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        clean = load_gray01(record.path)
        noisy = apply_noise(clean, record.relative)
        return torch.from_numpy(noisy).unsqueeze(0), torch.from_numpy(clean).unsqueeze(0)


def _loader(records, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(configs.SEED)
    return DataLoader(
        PairSet(records),
        batch_size=configs.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=worker_count(),
        generator=generator if shuffle else None,
    )


def _epoch_loss(model, loader, device, optimizer) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    seen = 0
    loss_fn = nn.MSELoss()
    with torch.set_grad_enabled(training):
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            if training:
                optimizer.zero_grad()
            loss = loss_fn(model(noisy), clean)
            if training:
                loss.backward()
                optimizer.step()
            total += float(loss.item()) * len(noisy)
            seen += len(noisy)
    return total / max(seen, 1)


def _check_watchdog(started: float) -> None:
    if time.perf_counter() - started > configs.WATCHDOG_SECONDS:
        raise RuntimeError("학습이 세 시간을 넘겼다. 배치 로더와 데이터 경로부터 확인한다")


def _fit(model, train_loader, val_loader, device, output_dir: Path) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.DENOISER_LR)
    best_loss = None
    best_state = None
    best_epoch = 0
    waited = 0
    started = time.perf_counter()
    for epoch in range(1, configs.DENOISER_EPOCHS + 1):
        train_loss = _epoch_loss(model, train_loader, device, optimizer)
        val_loss = _epoch_loss(model, val_loader, device, None)
        print(f"epoch {epoch} train {train_loss:.6f} val {val_loss:.6f}")
        if best_loss is None or val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            waited = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            waited += 1
        if best_state is not None:
            torch.save(
                {"state_dict": best_state, "best_val_loss": best_loss, "epoch": best_epoch},
                output_dir / "denoiser.pt",
            )
        _check_watchdog(started)
        if waited >= configs.DENOISER_PATIENCE:
            print("조기 종료")
            break
    model.load_state_dict(best_state)


def _save_strips(model, records, device, output_dir: Path) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda record: record.relative)[: configs.VAL_STRIP_COUNT]
    model.eval()
    maes = []
    with torch.no_grad():
        for index, record in enumerate(ordered, start=1):
            clean = load_gray01(record.path)
            noisy = apply_noise(clean, record.relative)
            gaussian = apply_gaussian_filter(noisy)
            tensor = torch.from_numpy(noisy).view(1, 1, *noisy.shape).to(device)
            residual = model(tensor).cpu().numpy()[0, 0]
            maes.append(float(np.mean(np.abs(residual - noisy))))
            panels = [clean, noisy, gaussian, residual, np.abs(residual - clean)]
            fig, axes = plt.subplots(1, len(panels), figsize=(15, 3))
            for axis, panel in zip(axes, panels):
                axis.imshow(panel, cmap="gray", vmin=0, vmax=1)
                axis.axis("off")
            fig.savefig(figure_dir / f"val_strip_{index:02d}.png", bbox_inches="tight")
            plt.close(fig)
    full_mae = _validation_mae(model, records, device)
    payload = {"val_mae_vs_noise": full_mae, "strip_mae_vs_noise": maes}
    (output_dir / "denoiser_mae.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _validation_mae(model, records, device) -> float:
    model.eval()
    total = 0.0
    seen = 0
    with torch.no_grad():
        for record in records:
            clean = load_gray01(record.path)
            noisy = apply_noise(clean, record.relative)
            tensor = torch.from_numpy(noisy).view(1, 1, *noisy.shape).to(device)
            residual = model(tensor).cpu().numpy()[0, 0]
            total += float(np.mean(np.abs(residual - noisy)))
            seen += 1
    return total / max(seen, 1)


def train_denoiser(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _root, split, train_records = require_gate(output_dir)
    train_set = records_for_patients(train_records, set(split["train_patients"]))
    val_set = records_for_patients(train_records, set(split["val_patients"]))
    sample = train_set[0]
    assert_noise_repeatable(load_gray01(sample.path), sample.relative)
    device = pick_device()
    set_seed()
    model = ResidualDenoiser().to(device)
    weight_path = output_dir / "denoiser.pt"
    if weight_path.exists():
        print("denoiser.pt가 있어 학습을 건너뛴다")
        model.load_state_dict(torch.load(weight_path, map_location=device)["state_dict"])
    else:
        _fit(model, _loader(train_set, True), _loader(val_set, False), device, output_dir)
    _save_strips(model, val_set, device, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    train_denoiser(Path(args.output_dir))


if __name__ == "__main__":
    main()
