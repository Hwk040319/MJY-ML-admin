# MJY-ML 운영진 전용 도구

**이 저장소는 절대 공개하지 마세요.** 비공개 정답과 채점 로직이 들어 있습니다.
참가자용 저장소는 `Hwk040319/MJY-ML` 입니다.

## 사전 준비 (배포 전)

```bash
# 1) 강의용 경량 샘플 생성 (experiment_id 단위로 클래스 균형 추출)
python make_lecture_subset.py \
    --data-root data_grouped \
    --output-dir data_lecture \
    --n-exp 4 --per-class 30

# 2) tar 로 묶어 Drive 에 업로드
tar -cf battery_lecture_sample.tar -C data_lecture train public_val
```

> 전체 배포본(`battery_train_val.tar`)은 `train`, `public_val` 두 폴더만 포함해야 합니다.
> 묶은 뒤 반드시 확인하세요.
> ```bash
> tar -tf battery_train_val.tar | cut -d/ -f1 | sort -u   # train, public_val 두 줄만 나와야 정상
> ```

## 팀별 Macro F1 확인 (2회차 전날)

**가장 쉬운 방법**: [`test_private.ipynb`](test_private.ipynb) 상단의 "Open in Colab" 배지를 눌러 열고,
안내에 따라 셀을 실행하면 됩니다. 팀의 `best_model.pt`를 업로드하면 바로 Macro F1이 나옵니다.
전체 절차(구글 드라이브 데이터 준비 등)는 [`채점_진행_가이드.md`](채점_진행_가이드.md)를 참고하세요.

로컬/터미널에서 직접 돌리고 싶다면 `score_checkpoint.py`를 씁니다:

```bash
python score_checkpoint.py \
    --checkpoint submissions/1조/best_model.pt \
    --data-root data_grouped
```

콘솔에 그 팀의 `Macro F1`, `Accuracy`, 클래스별 F1이 바로 출력됩니다.

> **⚠️ 최종 순위·점수 계산식은 아직 확정 전입니다.** 이전 버전의 `batch_score.py`에 있던
> "성능 70 × (팀 F1 ÷ 1위 F1) + 발표 30" 공식은 폐기되었습니다 — 확정된 공식이 아니었고
> 실제로 반영되지 않은 채 방치되어 혼란을 일으켰습니다. 공식이 확정되면 순위·리더보드를
> 만드는 스크립트를 다시 추가할 예정이며, 그 전까지는 팀별 Macro F1만 개별 확인하세요.

## 파일

| 파일 | 역할 |
|---|---|
| `make_lecture_subset.py` | 강의용 소규모 샘플 추출 (누수 없이) |
| `score_checkpoint.py` | 체크포인트 하나의 Macro F1/Accuracy/클래스별 F1 확인 |
| `test_private.ipynb` | 위 스크립트를 Colab에서 바로 실행하는 노트북 (Open in Colab) |
| `battery_dataset.py`, `common.py` | 참가자 저장소와 동일한 사본 (독립 실행용) |
| `채점_진행_가이드.md` | 운영진용 실행 가이드 (드라이브 준비 → Colab 실행까지) |

## 절대 하지 말 것

- 이 저장소를 public 으로 전환
- `private_labels.csv` 를 참가자 저장소나 Drive 공유 폴더에 두기
- `private_test` 가 포함된 원본 tar 를 공유 폴더에 두기
