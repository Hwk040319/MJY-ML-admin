"""2회차 — 팀별 제출 체크포인트를 모아 비공개 Test로 일괄 채점합니다.

배경:
    private_test 이미지는 학생에게 배포하지 않습니다. 따라서 학생은
    submission.csv 를 직접 만들 수 없고, 대신 best_model.pt 하나를 제출합니다.
    이 스크립트가 모든 팀의 체크포인트에 private_test 를 돌려 순위표를 만듭니다.

기대하는 제출 폴더 구조:
    submissions/
      1팀/best_model.pt
      2팀/best_model.pt
      ...

사용법:
    python batch_score.py \
        --submissions-dir submissions \
        --data-root data \
        --presentation-scores presentation_scores.csv \
        --output leaderboard.csv
"""

import argparse
import csv
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from battery_dataset import BatteryDataset
from common import NUM_CLASSES, get_transforms, resolve_device, set_seed


def load_model(checkpoint_path: Path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    image_size = ckpt.get("args", {}).get("image_size", 224)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), image_size


@torch.no_grad()
def predict(model, data_root, image_size, device, batch_size=64, num_workers=4):
    test_set = BatteryDataset(
        data_root, "private_test",
        transform=get_transforms(image_size, train=False),
        require_labels=False)
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    names, preds = [], []
    for images, _, batch_names in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            batch_preds = model(images).argmax(dim=1).cpu().tolist()
        names.extend(batch_names)
        preds.extend(batch_preds)
    return dict(zip(names, preds))


def score_against_labels(pred_map: dict, labels_path: Path):
    from sklearn.metrics import accuracy_score, f1_score

    truth = pd.read_csv(labels_path)  # image_name, target  (운영진만 보유)
    y_true, y_pred = [], []
    missing = 0
    for _, row in truth.iterrows():
        name = row["image_name"]
        if name not in pred_map:
            missing += 1
            continue
        y_true.append(int(row["target"]))
        y_pred.append(pred_map[name])

    return {
        "n_scored": len(y_true),
        "n_missing": missing,
        "accuracy": accuracy_score(y_true, y_pred) if y_true else 0.0,
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submissions-dir", required=True,
                    help="팀명 하위폴더에 best_model.pt 가 있는 상위 폴더")
    ap.add_argument("--data-root", required=True,
                    help="private_test/images 와 private_labels.csv 를 포함하는 폴더")
    ap.add_argument("--private-labels", default=None,
                    help="비공개 정답 CSV 경로 (기본: <data-root>/private_test/private_labels.csv)")
    ap.add_argument("--presentation-scores", default=None,
                    help="팀명,발표점수(30점 만점) 형식의 CSV. 없으면 발표점수 0으로 처리")
    ap.add_argument("--output", default="leaderboard.csv")
    args = ap.parse_args()

    set_seed(42)
    device = resolve_device()

    data_root = Path(args.data_root)
    labels_path = Path(args.private_labels) if args.private_labels else \
        data_root / "private_test" / "private_labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"비공개 정답 파일이 없습니다: {labels_path}")

    presentation = {}
    if args.presentation_scores:
        with open(args.presentation_scores, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                presentation[row["팀명"]] = float(row["발표점수"])

    sub_root = Path(args.submissions_dir)
    team_dirs = sorted(p for p in sub_root.iterdir() if p.is_dir())
    if not team_dirs:
        raise RuntimeError(f"{sub_root} 안에 팀 폴더가 없습니다.")

    results = []
    for team_dir in team_dirs:
        team = team_dir.name
        ckpt_path = team_dir / "best_model.pt"
        if not ckpt_path.is_file():
            print(f"[{team}] best_model.pt 없음 — 건너뜀 (0점 처리)")
            results.append({"team": team, "macro_f1": 0.0, "accuracy": 0.0,
                            "note": "체크포인트 없음"})
            continue

        try:
            model, image_size = load_model(ckpt_path, device)
            pred_map = predict(model, data_root, image_size, device)
            score = score_against_labels(pred_map, labels_path)
        except Exception as e:
            print(f"[{team}] 채점 실패: {e} — 0점 처리")
            results.append({"team": team, "macro_f1": 0.0, "accuracy": 0.0,
                            "note": f"오류: {e}"})
            continue

        print(f"[{team}] macro_f1={score['macro_f1']:.4f}  "
              f"accuracy={score['accuracy']:.4f}  "
              f"({score['n_scored']}장 채점, 누락 {score['n_missing']}장)")
        results.append({"team": team, "macro_f1": score["macro_f1"],
                        "accuracy": score["accuracy"], "note": ""})

    # ---------- 옵션 A: 70 × (팀 F1 / 1위 F1) + 발표 30 ----------
    best_f1 = max((r["macro_f1"] for r in results), default=0.0)
    for r in results:
        perf = 70.0 * (r["macro_f1"] / best_f1) if best_f1 > 0 else 0.0
        pres = presentation.get(r["team"], 0.0)
        r["performance_score"] = round(perf, 2)
        r["presentation_score"] = pres
        r["total_score"] = round(perf + pres, 2)

    # 동점 처리: 총점 -> macro_f1 -> accuracy 순으로 정렬
    results.sort(key=lambda r: (-r["total_score"], -r["macro_f1"], -r["accuracy"]))
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank

    out_path = Path(args.output)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "rank", "team", "macro_f1", "accuracy",
            "performance_score", "presentation_score", "total_score", "note"])
        w.writeheader()
        w.writerows(results)

    print("\n" + "=" * 70)
    print(f"{'순위':>4} {'팀':>10} {'MacroF1':>9} {'성능점수':>9} {'발표점수':>9} {'총점':>8}")
    for r in results:
        print(f"{r['rank']:>4} {r['team']:>10} {r['macro_f1']:>9.4f} "
              f"{r['performance_score']:>9.2f} {r['presentation_score']:>9.1f} "
              f"{r['total_score']:>8.2f}")
    print("=" * 70)
    print(f"저장: {out_path.resolve()}")


if __name__ == "__main__":
    main()
