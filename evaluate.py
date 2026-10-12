import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import sobel
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import confusion_matrix
from torchvision.models import resnet18
from torch import nn

import configs
from data import list_test_split, load_counts, load_gray01
from noise import apply_gaussian_filter, apply_noise
from train_classifier import macro_f1, pick_device
from train_denoiser import ResidualDenoiser


def _classifier(output_dir: Path, device: torch.device) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    checkpoint = torch.load(output_dir / "classifier.pt", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model


def _denoiser(output_dir: Path, device: torch.device):
    path = output_dir / "denoiser.pt"
    if not path.exists():
        return None
    model = ResidualDenoiser()
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _as_classifier_input(image: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).repeat(3, 1, 1)
    return tensor.unsqueeze(0).to(device)


def _predict(model, image: np.ndarray, device: torch.device) -> float:
    with torch.no_grad():
        logits = model(_as_classifier_input(image, device))
        return float(torch.softmax(logits, dim=1)[0, configs.LABEL_PNEUMONIA])


def _time_call(action, count: int, device: torch.device, use_cuda: bool) -> float:
    if use_cuda:
        _sync(device)
    started = time.perf_counter()
    action()
    if use_cuda:
        _sync(device)
    return (time.perf_counter() - started) * 1000 / count


def _sobel_corr(clean: np.ndarray, other: np.ndarray):
    left = np.hypot(sobel(clean, axis=0), sobel(clean, axis=1))
    right = np.hypot(sobel(other, axis=0), sobel(other, axis=1))
    if left.std() == 0 or right.std() == 0:
        return None
    return float(np.corrcoef(left.ravel(), right.ravel())[0, 1])


def _mean(values: list[float]) -> float | None:
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return None
    return float(np.mean(finite))


def _condition_summary(labels: np.ndarray, predictions: np.ndarray) -> dict:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "confusion": matrix.tolist(),
        "false_negative": int(matrix[1, 0]),
        "false_positive": int(matrix[0, 1]),
        "macro_f1": macro_f1(labels, predictions),
        "accuracy": float(np.mean(labels == predictions)),
    }


def _save_strip(path: Path, panels: list[np.ndarray]) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 3))
    for axis, panel in zip(axes, panels):
        axis.imshow(panel, cmap="gray", vmin=0, vmax=1)
        axis.axis("off")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _select_examples(rows: list[dict], has_residual: bool) -> dict[str, str | None]:
    by_file: dict[str, dict] = {}
    for row in rows:
        by_file.setdefault(row["file"], {})[row["condition"]] = row
    ordered = sorted(by_file, key=lambda name: (Path(name).name, name))

    def first(predicate):
        for name in ordered:
            if predicate(by_file[name]):
                return name
        return None

    def label_of(item):
        return item["clean"]["label"]

    def pred(item, condition):
        return item[condition]["prediction"]

    recovered = None
    still_missed = None
    if has_residual:
        recovered = first(
            lambda item: label_of(item) == 1 and pred(item, "clean") == 1 and pred(item, "noise") == 0 and pred(item, "residual") == 1
        )
        still_missed = first(lambda item: label_of(item) == 1 and pred(item, "residual") == 0)
    became_positive = first(
        lambda item: label_of(item) == 0
        and pred(item, "clean") == 0
        and any(pred(item, name) == 1 for name in item if name != "clean")
    )
    return {
        "test_correct.png": first(lambda item: label_of(item) == 1 and pred(item, "clean") == 1),
        "test_recovered.png": recovered,
        "test_still_missed.png": still_missed,
        "test_false_positive.png": became_positive,
    }


def _bars(path: Path, table: dict[str, dict]) -> None:
    names = list(table)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    metrics = (
        ("macro_f1", "macro-F1"),
        ("false_negative", "missed pneumonia"),
        ("false_positive", "normal as pneumonia"),
    )
    for axis, (key, title) in zip(axes, metrics):
        axis.bar(names, [table[name][key] for name in names])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_summary(path: Path, table, pixels, timing, examples, mae) -> None:
    lines = ["# 평가 요약", "", "숫자는 이 스크립트가 계산했다.", ""]
    for name, stats in table.items():
        lines.append(f"## {name}")
        lines.append(f"혼동행렬 [정상, 폐렴] 순: {stats['confusion']}")
        lines.append(f"폐렴 미탐 {stats['false_negative']}장, 정상 오탐 {stats['false_positive']}장")
        lines.append(f"macro-F1 {stats['macro_f1']:.4f}")
        lines.append("")
    lines.append("## 정확도 각주")
    for name, stats in table.items():
        lines.append(f"{name} 정확도 {stats['accuracy']:.4f}. 대표 점수는 macro-F1이다.")
    lines.append("")
    lines.append("## 픽셀")
    for name, stats in pixels.items():
        lines.append(
            f"{name} PSNR {_fmt(stats['psnr'])} 완전 일치 {stats['psnr_perfect']}장 "
            f"SSIM {_fmt(stats['ssim'])} 경계 상관 {_fmt(stats['sobel'])} "
            f"경계를 뺀 장수 {stats['sobel_skipped']}"
        )
    lines.append("")
    f1_best = max(table, key=lambda name: table[name]["macro_f1"])
    pixel_best = max(pixels, key=lambda name: pixels[name]["psnr"] if pixels[name]["psnr"] is not None else -1)
    same = "같다" if f1_best == pixel_best else "다르다"
    lines.append(f"macro-F1이 가장 높은 조건은 {f1_best}이다. PSNR이 가장 높은 조건은 {pixel_best}이다. 둘은 {same}.")
    lines.append("")
    lines.append("## 추론 시간")
    for name, item in timing.items():
        lines.append(f"{name} 이미지당 {item['ms']:.3f} ms, 기기 {item['device']}")
    lines.append("")
    lines.append("## 예시")
    for figure, chosen in examples.items():
        lines.append(f"{figure}: {chosen or '해당 유형 없음'}")
    if mae is not None:
        lines.append("")
        lines.append(f"검증에서 잔차망 출력과 노이즈 입력의 평균 절대 오차 {mae:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value) -> str:
    if value is None:
        return "없음"
    return f"{value:.4f}"


def evaluate(output_dir: Path) -> None:
    counts = load_counts(output_dir)
    if not counts.get("gate_passed"):
        raise RuntimeError("장수 또는 환자 키 확인을 통과하지 못했다")
    root = Path(counts["root"])
    records = list_test_split(root, output_dir)
    device = pick_device()
    classifier = _classifier(output_dir, device)
    denoiser = _denoiser(output_dir, device)
    threshold = json.loads((output_dir / "threshold.json").read_text(encoding="utf-8"))["threshold"]
    loaded = [{"record": record, "clean": load_gray01(record.path)} for record in records]
    for item in loaded:
        item["noise"] = apply_noise(item["clean"], item["record"].relative)

    def time_gaussian():
        for item in loaded:
            item["gaussian"] = apply_gaussian_filter(item["noise"])

    gaussian_ms = _time_call(time_gaussian, len(loaded), device, False)
    residual_ms = None
    if denoiser is not None:
        def time_residual():
            with torch.no_grad():
                for item in loaded:
                    tensor = torch.from_numpy(item["noise"]).view(1, 1, *item["noise"].shape).to(device)
                    item["residual"] = denoiser(tensor).cpu().numpy()[0, 0]

        residual_ms = _time_call(time_residual, len(loaded), device, True)

    def time_classifier():
        with torch.no_grad():
            for item in loaded:
                classifier(_as_classifier_input(item["clean"], device))

    classifier_ms = _time_call(time_classifier, len(loaded), device, device.type == "cuda")
    conditions = ["clean", "noise", "gaussian"] + (["residual"] if denoiser is not None else [])
    rows = []
    for item in loaded:
        for condition in conditions:
            probability = _predict(classifier, item[condition], device)
            prediction = int(probability >= threshold)
            rows.append(
                {
                    "file": item["record"].relative,
                    "label": item["record"].label,
                    "condition": condition,
                    "pneumonia_probability": probability,
                    "prediction": prediction,
                }
            )
    _write_csv(output_dir / "scores.csv", rows)
    table = {}
    pixels = {}
    by_condition: dict[str, list[dict]] = {name: [] for name in conditions}
    for row in rows:
        by_condition[row["condition"]].append(row)
    for condition in conditions:
        ordered = sorted(by_condition[condition], key=lambda row: row["file"])
        labels = np.asarray([row["label"] for row in ordered])
        predictions = np.asarray([row["prediction"] for row in ordered])
        table[condition] = _condition_summary(labels, predictions)
    for condition in ("noise", "gaussian", "residual"):
        if condition not in conditions:
            continue
        psnr_values, ssim_values, sobel_values = [], [], []
        skipped = 0
        for item in loaded:
            clean = item["clean"]
            other = item[condition]
            psnr_values.append(float(peak_signal_noise_ratio(clean, other, data_range=1)))
            ssim_values.append(float(structural_similarity(clean, other, data_range=1)))
            corr = _sobel_corr(clean, other)
            if corr is None:
                skipped += 1
            else:
                sobel_values.append(corr)
        pixels[condition] = {
            "psnr": _mean(psnr_values),
            "psnr_perfect": sum(not np.isfinite(value) for value in psnr_values),
            "ssim": _mean(ssim_values),
            "sobel": _mean(sobel_values),
            "sobel_skipped": skipped,
        }
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    lookup = {item["record"].relative: item for item in loaded}
    examples = _select_examples(rows, denoiser is not None)
    for figure, chosen in examples.items():
        if chosen is None:
            continue
        item = lookup[chosen]
        panels = [item["clean"], item["noise"], item["gaussian"]]
        if denoiser is not None:
            panels.extend([item["residual"], np.abs(item["residual"] - item["clean"])])
        _save_strip(figure_dir / figure, panels)
    _bars(figure_dir / "bars.png", table)
    device_name = "cpu" if device.type == "cpu" else torch.cuda.get_device_name(0)
    timing = {
        "classifier": {"ms": classifier_ms, "device": device_name},
        "gaussian": {"ms": gaussian_ms, "device": "cpu"},
    }
    if residual_ms is not None:
        timing["residual"] = {"ms": residual_ms, "device": device_name}
    mae_path = output_dir / "denoiser_mae.json"
    mae = json.loads(mae_path.read_text(encoding="utf-8"))["val_mae_vs_noise"] if mae_path.exists() else None
    _write_summary(output_dir / "summary.md", table, pixels, timing, examples, mae)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = ["file", "label", "condition", "pneumonia_probability", "prediction"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    evaluate(Path(args.output_dir))


if __name__ == "__main__":
    main()
