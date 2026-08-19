"""강의 실습용 소규모 데이터셋(기본 300장)을 원본 train/public_val에서 뽑습니다.

핵심 원칙: 이미지 단위가 아니라 experiment_id 단위로 뽑습니다.
같은 실험의 연속 프레임이 우연히 train 서브셋과 val 서브셋에 나뉘어 들어가면
작은 샘플 안에서도 데이터 누수가 생겨 '실습용인데 val 점수가 이상하게 높은'
상황이 발생합니다. 강의 내용(experiment_id 누수)과 직접 충돌하는 사고이므로
반드시 experiment_id 단위로 골라냅니다.

사용법:
    python make_lecture_subset.py \
        --data-root data_full \
        --output-dir data_lecture \
        --n-per-class 100
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd


def pick_experiments_for_target(df: pd.DataFrame, target: int, n_needed: int, seed: int):
    """target 클래스 이미지를 n_needed장 이상 채울 때까지 experiment_id를 통째로 뽑는다."""
    sub = df[df["target"] == target]
    exp_ids = sub["experiment_id"].drop_duplicates().sample(frac=1, random_state=seed).tolist()

    chosen_exps, collected = [], 0
    for exp_id in exp_ids:
        if collected >= n_needed:
            break
        chosen_exps.append(exp_id)
        collected += len(sub[sub["experiment_id"] == exp_id])

    return chosen_exps


def build_split(data_root: Path, split: str, out_root: Path, n_per_class: int, seed: int):
    src_dir = data_root / split
    df = pd.read_csv(src_dir / "labels.csv")

    if "experiment_id" not in df.columns:
        raise ValueError(f"{split}/labels.csv 에 experiment_id 열이 없습니다.")

    chosen_exps = set()
    for target in sorted(df["target"].unique()):
        chosen_exps |= set(pick_experiments_for_target(df, target, n_per_class, seed))

    picked = df[df["experiment_id"].isin(chosen_exps)].copy()

    out_dir = out_root / split
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    for name in picked["image_name"]:
        shutil.copy(src_dir / "images" / name, out_dir / "images" / name)
    picked.to_csv(out_dir / "labels.csv", index=False)

    counts = picked["target"].value_counts().sort_index().to_dict()
    print(f"[{split}] experiment_id {len(chosen_exps)}개 → 이미지 {len(picked)}장  "
          f"클래스분포 {counts}")
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="전체 train/public_val 이 있는 폴더")
    ap.add_argument("--output-dir", default="data_lecture")
    ap.add_argument("--n-per-class", type=int, default=100,
                    help="분할당 클래스별 최소 목표 장수 (기본 100 → 3클래스 약 300장)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.output_dir)

    for split in ["train", "public_val"]:
        build_split(data_root, split, out_root, args.n_per_class, args.seed)

    print(f"\n완료: {out_root.resolve()}")
    print("이 폴더를 zip으로 묶어 강의용 경량 Release 자산으로 올리세요.")
    print(f"  예: zip -r battery_lecture_sample.zip {out_root}")


if __name__ == "__main__":
    main()
