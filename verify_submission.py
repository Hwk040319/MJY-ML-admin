"""제출 전 자가 검증 — submission.csv 의 '형식'만 확인합니다.

정답을 사용하지 않으므로 점수는 알 수 없습니다.
형식 오류로 채점이 불가능해지는 상황을 미리 막는 것이 목적입니다.

사용법:
    python verify_submission.py --submission submission.csv --data-root data
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

VALID = {0, 1, 2}
EXPECTED_COLUMNS = ["image_name", "predicted_class"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="submission.csv")
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()

    problems = []
    sub_path = Path(args.submission)
    if not sub_path.is_file():
        print(f"파일이 없습니다: {sub_path.resolve()}")
        sys.exit(1)

    df = pd.read_csv(sub_path)

    # 1) 열 이름과 순서
    if list(df.columns) != EXPECTED_COLUMNS:
        problems.append(
            f"열이 {list(df.columns)} 입니다. {EXPECTED_COLUMNS} 여야 합니다. "
            f"(index=False 로 저장했는지 확인하세요)")
        print("열 이름이 달라 이후 검사를 진행할 수 없습니다.")
        for m in problems:
            print("  - " + m)
        sys.exit(1)

    # 2) 예측값 범위
    bad = set(df["predicted_class"].unique()) - VALID
    if bad:
        problems.append(f"predicted_class 에 허용되지 않은 값: {sorted(bad)}")

    # 3) 중복
    dup = df["image_name"].duplicated().sum()
    if dup:
        problems.append(f"image_name 중복 {dup}건")

    # 4) 결측
    if df.isna().any().any():
        problems.append("빈 칸(NaN)이 포함되어 있습니다")

    # 5) 실제 private_test 이미지 목록과 대조
    image_dir = Path(args.data_root) / "private_test" / "images"
    if image_dir.is_dir():
        expected = {p.name for p in image_dir.glob("*.png")}
        got = set(df["image_name"])
        missing, extra = expected - got, got - expected
        if missing:
            problems.append(f"예측이 빠진 이미지 {len(missing)}장 (예: {sorted(missing)[:3]})")
        if extra:
            problems.append(f"존재하지 않는 이미지 {len(extra)}건 (예: {sorted(extra)[:3]})")
        print(f"private_test 이미지 {len(expected)}장 / 제출 {len(df)}행")
    else:
        print(f"[안내] {image_dir} 를 찾지 못해 행 수 대조는 건너뜁니다.")
        print(f"제출 {len(df)}행")

    counts = df["predicted_class"].value_counts().sort_index()
    print("예측 분포: " + "  ".join(f"{k}={v}" for k, v in counts.items()))

    print("-" * 60)
    if problems:
        print(f"형식 문제 {len(problems)}건. 수정 후 다시 확인하세요.\n")
        for m in problems:
            print("  - " + m)
        sys.exit(1)

    print("형식 검사 통과. 제출해도 좋습니다.")


if __name__ == "__main__":
    main()
