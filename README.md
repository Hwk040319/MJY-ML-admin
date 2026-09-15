# MJY-ML 운영진 전용 도구

**⚠️ 현재 이 저장소는 public 상태입니다.** 누구나 코드를 볼 수 있습니다.
다만 정답 파일(`private_labels.csv`)과 `private_test` 원본 이미지는 이 저장소에 들어있지 않고
운영진이 별도 경로로만 전달하므로, 코드가 공개되어 있어도 정답 자체가 저장소를 통해 유출되지는 않습니다.
그래도 앞으로 파일을 추가/수정할 때 정답·개인정보가 절대 커밋되지 않도록 주의하세요.
되돌리려면(Private 전환) 저장소 Settings에서 소유자 계정으로 직접 변경하면 됩니다.
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
전체 절차(정답 파일 전달받기, 드라이브 데이터 준비 등)는 [`채점_진행_가이드.md`](채점_진행_가이드.md)를 참고하세요.

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
| `채점_진행_가이드.md` | 운영진용 실행 가이드 (정답 파일 전달받기 → Colab 실행까지) |

## 절대 하지 말 것

- `private_labels.csv`를 이 저장소, 참가자 저장소, 또는 아무 Drive 공유 폴더에도 커밋/업로드하지 말 것 — 운영진 간 직접 전달(카톡/이메일 등)로만 공유
- `private_test`가 포함된 원본 tar를 참가자와 공유하는 폴더에 두기
- (이미 public이 된 뒤로는 특히) 정답이나 채점 대상 데이터를 이 저장소에 커밋하기
