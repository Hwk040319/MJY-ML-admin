"""2회차 — 잠근 모델로 비공개 Test 를 예측해 submission.csv 를 만듭니다.

주의:
    이 스크립트는 '최종 모델을 고른 뒤' 한 번만 실행합니다.
    비공개 Test 점수를 보고 다시 학습하면 최종 평가의 독립성이 깨집니다.

사용법:
    python predict_test.py --data-root data \
        --checkpoint outputs/exp_aug/best_model.pt \
        --output submission.csv
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from battery_dataset import BatteryDataset
from common import NUM_CLASSES, get_transforms, resolve_device, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default="submission.csv")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    set_seed(42)
    device = resolve_device()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"체크포인트가 없습니다: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})
    image_size = train_args.get("image_size", 224)

    print(f"체크포인트: {ckpt_path}")
    print(f"  학습 epoch {ckpt.get('epoch')}, Public Val Macro F1 {ckpt.get('macro_f1'):.4f}")
    print(f"  입력 크기 {image_size} (학습 때와 동일하게 맞춥니다)")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()

    # private_test 는 labels.csv 가 없으므로 require_labels=False
    test_set = BatteryDataset(
        args.data_root, "private_test",
        transform=get_transforms(image_size, train=False),
        require_labels=False)
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    rows = []
    with torch.no_grad():
        for images, _, names in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                preds = model(images).argmax(dim=1).cpu().tolist()
            rows.extend(zip(names, preds))

    out_path = Path(args.output)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_name", "predicted_class"])
        w.writerows(rows)

    print(f"\n{len(rows)}행을 {out_path.resolve()} 에 저장했습니다.")
    print("제출 전에 반드시 실행하세요:")
    print(f"  python verify_submission.py --submission {out_path} --data-root {args.data_root}")


if __name__ == "__main__":
    main()
