"""강의 실습용 소규모 데이터셋을 원본 train/public_val에서 뽑습니다.

핵심 원칙 두 가지

1. experiment_id 단위로 실험을 고른다
   같은 실험의 연속 프레임이 train 서브셋과 val 서브셋에 나뉘어 들어가면
   작은 샘플 안에서도 데이터 누수가 생깁니다. 강의에서 다루는 내용과
   정면으로 충돌하는 사고이므로 실험 단위로 먼저 나눕니다.

2. 실험 안에서 클래스별로 정해진 장수만 샘플링한다
   experiment_id 하나가 1,000장을 넘는 경우가 많습니다. 실험을 통째로
   가져오면 목표 장수를 크게 초과하고 클래스 균형도 무너집니다.
   그래서 실험을 여러 개 고른 뒤, 각 실험 안에서 클래스마다
   --per-class 장씩만 뽑습니다.

장수 계산법
    분할당 = n_exp x 클래스 3개 x per_class
    전체   = 분할당 x 2 (train + public_val)

    기본값(--n-exp 4 --per-class 30) 기준
        분할당 최대 360장, 전체 최대 720장 (실측 약 690장)
    전체를 약 350장으로 줄이려면
        --per-class 15  또는  --n-exp 2

사용 예:
    python make_lecture_subset.py \
        --data-root data_grouped \
        --output-dir data_lecture \
        --n-exp 4 --per-class 30
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd


def build_split(data_root: Path, split: str, out_root: Path,
                n_exp: int, per_class: int, seed: int):
    src = data_root / split
    df = pd.read_csv(src / "labels.csv")

    if "experiment_id" not in df.columns:
        raise ValueError(f"{split}/labels.csv 에 experiment_id 열이 없습니다.")

    n_available = df["experiment_id"].nunique()
    exps = (df["experiment_id"].drop_duplicates()
            .sample(n=min(n_exp, n_available), random_state=seed).tolist())

    picked = []
    for e in exps:
        sub = df[df["experiment_id"] == e]
        for t in sorted(df["target"].unique()):
            cls = sub[sub["target"] == t]
            if len(cls) == 0:
                continue
            picked.append(cls.sample(n=min(per_class, len(cls)),
                                     random_state=seed))
    picked = pd.concat(picked)

    out = out_root / split
    (out / "images").mkdir(parents=True, exist_ok=True)
    for name in picked["image_name"]:
        shutil.copy(src / "images" / name, out / "images" / name)
    picked.to_csv(out / "labels.csv", index=False)

    counts = picked["target"].value_counts().sort_index().to_dict()
    print(f"[{split}] 실험 {len(exps)}개 -> 이미지 {len(picked)}장  클래스분포 {counts}")
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="전체 train/public_val 이 있는 폴더")
    ap.add_argument("--output-dir", default="data_lecture")
    ap.add_argument("--n-exp", type=int, default=4,
                    help="분할별로 고를 experiment_id 개수 (기본 4)")
    ap.add_argument("--per-class", type=int, default=30,
                    help="실험 하나 안에서 클래스별로 뽑을 장수 (기본 30). "
                         "전체 장수 = n_exp x 3 x per_class x 2")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.output_dir)

    total = 0
    for split in ["train", "public_val"]:
        picked = build_split(data_root, split, out_root,
                             args.n_exp, args.per_class, args.seed)
        total += len(picked)

    print(f"\n전체 {total}장 (train + public_val 합계)")
    print(f"완료: {out_root.resolve()}")
    print("tar 로 묶어 Drive 에 올리세요:")
    print(f"  tar -cf battery_lecture_sample.tar -C {out_root} train public_val")
    print("묶은 뒤 반드시 확인 (train, public_val 두 줄만 나와야 정상):")
    print("  tar -tf battery_lecture_sample.tar | cut -d/ -f1 | sort -u")


if __name__ == "__main__":
    main()
