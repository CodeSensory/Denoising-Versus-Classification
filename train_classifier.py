import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

import configs
from data import load_gray01, records_for_patients, require_gate, worker_count


class CleanSet(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        gray = load_gray01(record.path)
        tensor = torch.from_numpy(gray).unsqueeze(0).repeat(3, 1, 1)
        return tensor, record.label


def set_seed() -> None:
    np.random.seed(configs.SEED)
    torch.manual_seed(configs.SEED)
    torch.cuda.manual_seed_all(configs.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def write_environment(output_dir: Path, device: torch.device) -> None:
    import skimage

    import torchvision

    name = "cpu" if device.type == "cpu" else torch.cuda.get_device_name(0)
    payload = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "skimage": skimage.__version__,
        "device": name,
    }
    path = output_dir / "environment.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_classifier() -> nn.Module:
    try:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except Exception as error:
        raise RuntimeError("ImageNet 가중치를 받지 못했다. Internet을 켜고 이 단계만 다시 실행한다") from error
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(f1_score(labels, predictions, average="macro", labels=[0, 1], zero_division=0))


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    candidates = []
    best_score = None
    best_threshold = None
    steps = int(round((configs.THRESHOLD_STOP - configs.THRESHOLD_START) / configs.THRESHOLD_STEP)) + 1
    grid = [round(configs.THRESHOLD_START + configs.THRESHOLD_STEP * index, 2) for index in range(steps)]
    for threshold in grid:
        predictions = (probabilities >= threshold).astype(np.int64)
        score = macro_f1(labels, predictions)
        candidates.append({"threshold": threshold, "macro_f1": score})
        if best_score is None or score > best_score:
            best_score = score
            best_threshold = threshold
    return {
        "threshold": best_threshold,
        "val_macro_f1": best_score,
        "early_stop": "조기 종료는 폐렴 확률이 0.5를 넘을 때의 검증 macro-F1",
        "candidates": candidates,
    }


def _loader(records, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(configs.SEED)
    return DataLoader(
        CleanSet(records),
        batch_size=configs.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=worker_count(),
        generator=generator if shuffle else None,
    )


def _run_epoch(model, loader, device, optimizer) -> float:
    model.train()
    total = 0.0
    seen = 0
    loss_fn = nn.CrossEntropyLoss()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * len(labels)
        seen += len(labels)
    return total / max(seen, 1)


def _collect(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_out = []
    probs_out = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)[:, configs.LABEL_PNEUMONIA]
            labels_out.extend(labels.tolist())
            probs_out.extend(probs.cpu().tolist())
    return np.asarray(labels_out), np.asarray(probs_out)


def _check_watchdog(started: float) -> None:
    if time.perf_counter() - started > configs.WATCHDOG_SECONDS:
        raise RuntimeError("학습이 세 시간을 넘겼다. 배치 로더와 데이터 경로부터 확인한다")


def _fit(model, train_loader, val_loader, device, output_dir: Path) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.CLASSIFIER_LR)
    best_score = -1.0
    best_state = None
    best_epoch = 0
    waited = 0
    started = time.perf_counter()
    for epoch in range(1, configs.CLASSIFIER_EPOCHS + 1):
        loss = _run_epoch(model, train_loader, device, optimizer)
        labels, probs = _collect(model, val_loader, device)
        score = macro_f1(labels, (probs > 0.5).astype(np.int64))
        print(f"epoch {epoch} loss {loss:.4f} val macro-F1 {score:.4f}")
        if score > best_score:
            best_score = score
            best_epoch = epoch
            waited = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            waited += 1
        if best_state is not None:
            torch.save(
                {"state_dict": best_state, "best_macro_f1": best_score, "epoch": best_epoch},
                output_dir / "classifier.pt",
            )
        _check_watchdog(started)
        if waited >= configs.CLASSIFIER_PATIENCE:
            print("조기 종료")
            break
    model.load_state_dict(best_state)


def _write_threshold(model, val_loader, device, output_dir: Path) -> None:
    labels, probs = _collect(model, val_loader, device)
    payload = choose_threshold(labels, probs)
    path = output_dir / "threshold.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"문턱값 {payload['threshold']} 검증 macro-F1 {payload['val_macro_f1']:.4f}")


def train_classifier(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _root, split, train_records = require_gate(output_dir)
    train_set = records_for_patients(train_records, set(split["train_patients"]))
    val_set = records_for_patients(train_records, set(split["val_patients"]))
    device = pick_device()
    set_seed()
    write_environment(output_dir, device)
    model = build_classifier().to(device)
    weight_path = output_dir / "classifier.pt"
    if weight_path.exists():
        print("classifier.pt가 있어 학습을 건너뛴다")
        model.load_state_dict(torch.load(weight_path, map_location=device)["state_dict"])
    else:
        _fit(model, _loader(train_set, True), _loader(val_set, False), device, output_dir)
    if (output_dir / "threshold.json").exists():
        print("threshold.json이 이미 있다")
        return
    _write_threshold(model, _loader(val_set, False), device, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    train_classifier(Path(args.output_dir))


if __name__ == "__main__":
    main()
