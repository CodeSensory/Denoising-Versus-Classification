import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import configs

PERSON_RE = re.compile(r"(person\d+)", re.IGNORECASE)
IM_RE = re.compile(r"((?:NORMAL2-)?IM-\d+)", re.IGNORECASE)
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}
SPLIT_NAMES = ("train", "test", "val")
CLASS_NAMES = ("NORMAL", "PNEUMONIA")


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    relative: str
    label: int
    label_name: str
    patient: str
    split: str


def patient_key(stem: str) -> str | None:
    person = PERSON_RE.search(stem)
    if person:
        return person.group(1).lower()
    image_id = IM_RE.search(stem)
    if image_id:
        return image_id.group(1).upper()
    return None


def find_chest_root(search_root: Path) -> tuple[Path, list[str]]:
    found = []
    skipped_oct = []
    for dirpath, dirnames, _ in os.walk(search_root):
        kept = []
        for name in dirnames:
            upper = name.upper()
            if upper == "OCT" or upper == "__MACOSX" or name.startswith("."):
                skipped_oct.append(str(Path(dirpath) / name))
                continue
            kept.append(name)
        dirnames[:] = kept
        path = Path(dirpath)
        if _is_chest_root(path):
            found.append(path)
    if not found:
        raise FileNotFoundError(f"train/test/val 흉부 영상 루트를 찾지 못했다: {search_root}")
    found = [path for path in found if "__MACOSX" not in path.parts]
    if not found:
        raise FileNotFoundError(f"__MACOSX를 제외한 흉부 영상 루트가 없다: {search_root}")
    found.sort(key=lambda path: (-_count_images(path / "train"), len(path.parts)))
    return found[0], skipped_oct


def _is_chest_root(path: Path) -> bool:
    for split in SPLIT_NAMES:
        for class_name in CLASS_NAMES:
            if not (path / split / class_name).is_dir():
                return False
    return True


def _count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES)


def list_split(root: Path, split: str) -> list[ImageRecord]:
    if split not in ("train", "val"):
        raise RuntimeError("학습 쪽 목록은 train과 공식 val만 연다")
    return _list_split(root, split)


def list_test_split(root: Path, output_dir: Path) -> list[ImageRecord]:
    threshold = output_dir / "threshold.json"
    classifier = output_dir / "classifier.pt"
    if not threshold.exists() or not classifier.exists():
        raise RuntimeError("threshold.json과 classifier.pt가 있을 때만 공식 테스트를 연다")
    return _list_split(root, "test")


def _list_split(root: Path, split: str) -> list[ImageRecord]:
    records = []
    for class_name in CLASS_NAMES:
        folder = root / split / class_name
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.name.startswith("._") or path.name.startswith("."):
                continue
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            key = patient_key(path.stem)
            relative = path.relative_to(root).as_posix()
            records.append(
                ImageRecord(
                    path=path,
                    relative=relative,
                    label=configs.CLASS_TO_LABEL[class_name],
                    label_name=class_name,
                    patient=key or "",
                    split=split,
                )
            )
    return records


def load_gray01(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    width, height = image.size
    short = min(width, height)
    scale = configs.IMAGE_SIZE / short
    new_width = max(configs.IMAGE_SIZE, int(round(width * scale)))
    new_height = max(configs.IMAGE_SIZE, int(round(height * scale)))
    resample = getattr(Image, "Resampling", Image).BILINEAR
    image = image.resize((new_width, new_height), resample)
    left = (new_width - configs.IMAGE_SIZE) // 2
    top = (new_height - configs.IMAGE_SIZE) // 2
    image = image.crop((left, top, left + configs.IMAGE_SIZE, top + configs.IMAGE_SIZE))
    return np.asarray(image, dtype=np.float32) / np.float32(255)


def worker_count() -> int:
    if os.name == "nt":
        return 0
    return configs.NUM_WORKERS


def split_patients(train_records: list[ImageRecord]) -> tuple[list[str], list[str]]:
    by_label: dict[str, list[str]] = {name: [] for name in CLASS_NAMES}
    seen: dict[str, str] = {}
    for record in train_records:
        if not record.patient:
            continue
        if record.patient in seen and seen[record.patient] != record.label_name:
            raise RuntimeError(f"한 환자 키가 두 클래스에 있다: {record.patient}")
        seen[record.patient] = record.label_name
    for patient, label_name in seen.items():
        by_label[label_name].append(patient)
    rng = np.random.default_rng(configs.SEED)
    val_patients: list[str] = []
    train_patients: list[str] = []
    for label_name in CLASS_NAMES:
        keys = sorted(set(by_label[label_name]))
        if not keys:
            raise RuntimeError(f"학습 폴더에 {label_name} 환자가 없다")
        n_val = max(1, int(round(configs.VAL_FRACTION * len(keys))))
        picked = set(rng.permutation(len(keys))[:n_val].tolist())
        for index, key in enumerate(keys):
            if index in picked:
                val_patients.append(key)
            else:
                train_patients.append(key)
    overlap = set(train_patients) & set(val_patients)
    if overlap:
        raise RuntimeError(f"학습과 검증에 같이 들어간 환자: {sorted(overlap)[:5]}")
    if not train_patients:
        raise RuntimeError("학습 환자가 비었다")
    return sorted(train_patients), sorted(val_patients)


def records_for_patients(train_records: list[ImageRecord], patients: set[str]) -> list[ImageRecord]:
    chosen = [record for record in train_records if record.patient in patients]
    for record in chosen:
        if not record.relative.startswith("train/"):
            raise RuntimeError("학습 함수가 테스트 경로를 받았다")
    if not chosen:
        raise RuntimeError("환자 목록에 해당하는 학습 영상이 없다")
    return chosen


def inspect_dataset(search_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root, skipped_oct = find_chest_root(search_root)
    grouped = {split: _list_split(root, split) for split in SPLIT_NAMES}
    _write_samples(output_dir / "filename_samples.txt", grouped)
    unmatched = [record.relative for records in grouped.values() for record in records if not record.patient]
    dual = _dual_label_patients(grouped)
    overlap = _train_test_overlap(grouped)
    counts = _counts_payload(root, grouped, unmatched, dual, overlap, skipped_oct)
    _write_json(output_dir / "counts.json", counts)
    if unmatched or dual or not counts["gate_passed"]:
        reasons = []
        if counts["folders"]["train"]["total"] != configs.EXPECTED_COUNTS["train"]:
            reasons.append(f"학습 {counts['folders']['train']['total']}장")
        if counts["folders"]["test"]["total"] != configs.EXPECTED_COUNTS["test"]:
            reasons.append(f"테스트 {counts['folders']['test']['total']}장")
        if counts["folders"]["val"]["total"] != configs.EXPECTED_COUNTS["val"]:
            reasons.append(f"공식 검증 {counts['folders']['val']['total']}장")
        if unmatched:
            reasons.append(f"환자 키가 없는 파일 {len(unmatched)}개")
        if dual:
            reasons.append(f"두 클래스에 걸친 환자 {dual[:5]}")
        raise RuntimeError("학습을 시작하지 않는다. " + ", ".join(reasons))
    train_patients, val_patients = split_patients(grouped["train"])
    _write_json(
        output_dir / "split.json",
        {
            "seed": configs.SEED,
            "val_fraction": configs.VAL_FRACTION,
            "rule": configs.PATIENT_RULE,
            "train_patients": train_patients,
            "val_patients": val_patients,
        },
    )


def _dual_label_patients(grouped: dict[str, list[ImageRecord]]) -> list[str]:
    labels: dict[str, set[str]] = {}
    for records in grouped.values():
        for record in records:
            if record.patient:
                labels.setdefault(record.patient, set()).add(record.label_name)
    return sorted(key for key, names in labels.items() if len(names) > 1)


def _train_test_overlap(grouped: dict[str, list[ImageRecord]]) -> int:
    train_keys = {record.patient for record in grouped["train"] if record.patient}
    test_keys = {record.patient for record in grouped["test"] if record.patient}
    return len(train_keys & test_keys)


def _counts_payload(root, grouped, unmatched, dual, overlap, skipped_oct) -> dict:
    folders = {}
    for split, records in grouped.items():
        folders[split] = {
            "NORMAL": sum(record.label_name == "NORMAL" for record in records),
            "PNEUMONIA": sum(record.label_name == "PNEUMONIA" for record in records),
            "total": len(records),
            "patients": len({record.patient for record in records if record.patient}),
        }
    expected_ok = all(folders[split]["total"] == configs.EXPECTED_COUNTS[split] for split in SPLIT_NAMES)
    return {
        "root": str(root),
        "oct_skipped": skipped_oct,
        "folders": folders,
        "train_test_patient_overlap": overlap,
        "unmatched_files": unmatched,
        "dual_label_patients": dual,
        "gate_passed": expected_ok and not unmatched and not dual,
    }


def _write_samples(path: Path, grouped: dict[str, list[ImageRecord]]) -> None:
    lines = []
    for split in SPLIT_NAMES:
        for class_name in CLASS_NAMES:
            names = sorted(record.relative for record in grouped[split] if record.label_name == class_name)
            lines.append(f"[{split}/{class_name}]")
            lines.extend(names[: configs.SAMPLE_PER_CLASS])
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_counts(output_dir: Path) -> dict:
    path = output_dir / "counts.json"
    if not path.exists():
        raise RuntimeError("counts.json이 없다. 1일 차 확인부터 실행한다")
    return json.loads(path.read_text(encoding="utf-8"))


def load_split(output_dir: Path) -> dict:
    path = output_dir / "split.json"
    if not path.exists():
        raise RuntimeError("split.json이 없다. 환자 규칙을 확인한 뒤에 학습한다")
    return json.loads(path.read_text(encoding="utf-8"))


def require_gate(output_dir: Path) -> tuple[Path, dict, list[ImageRecord]]:
    counts = load_counts(output_dir)
    if not counts.get("gate_passed"):
        raise RuntimeError("장수 또는 환자 키 확인을 통과하지 못했다")
    split = load_split(output_dir)
    root = Path(counts["root"])
    train_records = list_split(root, "train")
    return root, split, train_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    inspect_dataset(Path(args.search_root), Path(args.output_dir))


if __name__ == "__main__":
    main()
