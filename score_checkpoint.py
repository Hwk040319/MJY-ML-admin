"""체크포인트 하나를 비공개 Test로 채점해 Macro F1만 바로 확인합니다.

점수 공식·순위 계산 없이 Macro F1/Accuracy/클래스별 F1만 출력합니다.
채점 공식이 아직 확정 전이라, 팀별 Macro F1만 빠르게 확인하고 싶을 때 씁니다.

사용법:
    python score_checkpoint.py --checkpoint best_model.pt --data-root data_grouped
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from battery_dataset import BatteryDataset
from common import CLASS_NAMES, NUM_CLASSES, get_transforms, resolve_device, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="채점할 best_model.pt 경로")
    ap.add_argument("--data-root", required=True,
                    help="private_test/images 와 private_labels.csv 를 포함하는 폴더")
    ap.add_argument("--private-labels", default=None,
                    help="비공개 정답 CSV 경로 (기본: <data-root>/private_test/private_labels.csv)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    set_seed(42)
    device = resolve_device()

    data_root = Path(args.data_root)
    labels_path = Path(args.private_labels) if args.private_labels else \
        data_root / "private_test" / "private_labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"비공개 정답 파일이 없습니다: {labels_path}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"체크포인트가 없습니다: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    image_size = ckpt.get("args", {}).get("image_size", 224)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()

    test_set = BatteryDataset(
        data_root, "private_test",
        transform=get_transforms(image_size, train=False),
        require_labels=False)
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    names, preds = [], []
    with torch.no_grad():
        for images, _, batch_names in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                batch_preds = model(images).argmax(dim=1).cpu().tolist()
            names.extend(batch_names)
            preds.extend(batch_preds)
    pred_map = dict(zip(names, preds))

    from sklearn.metrics import accuracy_score, f1_score

    truth = pd.read_csv(labels_path)  # image_name, target (운영진만 보유)
    y_true, y_pred = [], []
    missing = 0
    for _, row in truth.iterrows():
        name = row["image_name"]
        if name not in pred_map:
            missing += 1
            continue
        y_true.append(int(row["target"]))
        y_pred.append(pred_map[name])

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    class_f1 = f1_score(y_true, y_pred, average=None,
                         labels=list(range(NUM_CLASSES)), zero_division=0) if y_true else [0, 0, 0]

    print("=" * 50)
    print(f"체크포인트: {ckpt_path.name}")
    print(f"학습 epoch {ckpt.get('epoch')}, 학습 당시 Public Val Macro F1 {ckpt.get('macro_f1')}")
    print(f"채점 이미지 수: {len(y_true)}장 (누락 {missing}장)")
    print(f"Macro F1 : {macro_f1:.4f}")
    print(f"Accuracy : {acc:.4f}")
    print("Class F1 : " + "  ".join(f"{n} {v:.4f}" for n, v in zip(CLASS_NAMES, class_f1)))
    print("=" * 50)


if __name__ == "__main__":
    main()
