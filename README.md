# MJY-ML 운영진 전용 도구

**이 저장소는 절대 공개하지 마세요.** 비공개 정답과 채점 로직이 들어 있습니다.
참가자용 저장소는 `Hwk040319/MJY-ML` 입니다.

## 사전 준비 (배포 전)

```bash
# 1) 강의용 경량 샘플 생성 (experiment_id 단위로 클래스 균형 추출)
python make_lecture_subset.py \
    --data-root data_grouped \
    --output-dir data_lecture \
    --n-per-class 100

# 2) tar 로 묶어 Drive 에 업로드
tar -cf battery_lecture_sample.tar -C data_lecture train public_val
```

> 전체 배포본(`battery_train_val.tar`)은 `train`, `public_val` 두 폴더만 포함해야 합니다.
> 묶은 뒤 반드시 확인하세요.
> ```bash
> tar -tf battery_train_val.tar | cut -d/ -f1 | sort -u   # train, public_val 두 줄만 나와야 정상
> ```

## 채점 (2회차 전날)

제출받은 체크포인트를 팀별 폴더에 정리합니다.

```
submissions/
  1조/best_model.pt
  2조/best_model.pt
  ...
```

발표 점수는 CSV 로 준비합니다 (`팀명,발표점수` 헤더, 30점 만점).

```bash
python batch_score.py \
    --submissions-dir submissions \
    --data-root data_grouped \
    --private-labels data_grouped/private_test/private_labels.csv \
    --presentation-scores presentation_scores.csv \
    --output leaderboard.csv
```

순위 계산식은 `성능 70 × (팀 F1 ÷ 1위 F1) + 발표 30` 이며,
동점 시 Macro F1 → Accuracy 순으로 정렬됩니다.

## 파일

| 파일 | 역할 |
|---|---|
| `make_lecture_subset.py` | 강의용 소규모 샘플 추출 (누수 없이) |
| `batch_score.py` | 팀별 체크포인트 일괄 채점 + 리더보드 생성 |
| `predict_test.py` | 단일 체크포인트 예측 (디버깅용) |
| `verify_submission.py` | CSV 형식 검증 (예비용) |
| `battery_dataset.py`, `common.py` | 참가자 저장소와 동일한 사본 (독립 실행용) |

## 절대 하지 말 것

- 이 저장소를 public 으로 전환
- `private_labels.csv` 를 참가자 저장소나 Drive 공유 폴더에 두기
- `private_test` 가 포함된 원본 tar 를 공유 폴더에 두기
