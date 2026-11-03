# EmbeddingGemma 2 멀티모달 임베딩 실습 (Jetson Orin Nano Super)

정원일 · AIGJ_01_014

**결과 보고서: [`report.md`](report.md)** — 6~9단계 결과표, 10단계 진행 중

## 폴더 구성

| 폴더 | 단계 | 내용 |
| --- | --- | --- |
| [`common/`](common) | 1~2 | 공통 로더 `eg2.py`(`load()`), GPU 확인 `check.py` |
| [`stage03-05_embedding_basics/`](stage03-05_embedding_basics) | 3~5 | 텍스트·비전·오디오 검색, 음성 샘플 생성, `media/` 합성 음성·소리 |
| [`stage06_performance/`](stage06_performance) | 6 | 구성별 RAM·지연·GPU 측정, `results/` 측정 로그 |
| [`stage07_finetune/`](stage07_finetune) | 7 | 도메인 미세 조정, 검색 평가, `data/` 학습·평가 쌍, `results/` |
| [`stage08_object_recognition/`](stage08_object_recognition) | 8 | 웹캠 물체 검색 웹 앱(yolo / relabel / yoloe), `results/` |
| [`stage09_pose_behavior/`](stage09_pose_behavior) | 9 | 동작·행동 인식, GRU 학습(`models/`), 라벨(`data/`), `results/` |
| [`stage10_proximity/`](stage10_proximity) | 10 | 카메라 2대 접근 감지(맥북+아이폰), 보정 스크립트·`calib/` |
| [`tools/`](tools) | - | 6~9단계 표 자동 정리 |

## 실행 방법

보드의 venv(`source ~/eg2/bin/activate`, torch·sentence-transformers·ultralytics)에서 **각 단계 폴더로 들어가 실행**한다. 공통 로더 `common/eg2.py`는 스크립트가 알아서 찾는다. `run_*.sh`는 어디서 불러도 자기 폴더로 이동한다.

| 단계 | 실행 | 결과 |
| --- | --- | --- |
| 3~5 | `cd stage03-05_embedding_basics && python test_text.py` (`test_vision.py`, `test_audio.py`) | 화면 출력 |
| 6~7 | `stage06_performance/run_stage67.sh` | `stage06_performance/results/`, `stage07_finetune/results/` |
| 7 학습 | `cd stage07_finetune && python train_ft.py` → `models/eg2-classroom` (약 1GB, 저장소에 없음) | |
| 8 | `stage08_object_recognition/run_stage8.sh` (입력 `src/` 물품 사진 5장) | `results/<mode>.json` |
| 9 | `stage09_pose_behavior/run_stage9.sh` → `python label_feat.py` → `python train_pose.py ...` → `run_stage9_model.sh` | `results/`, `data/`, `models/` |
| 10 | `cd stage10_proximity` 후 `mac_stream.sh`(맥) → `calib_intrinsics.py` → `calib_extrinsics.py` → `proximity_app.py` | `calib/` |
| 표 정리 | `python tools/summarize_results.py` | 보고서 6~9단계 표 |

- 9단계 입력 영상은 사람이 나와서 올리지 않았다. 목록은 [`stage09_pose_behavior/videos.md`](stage09_pose_behavior/videos.md)에 있다(Pexels 번호, 같은 이름으로 `src_pose/`·`src_pose_new/`에 두면 그대로 돈다).
- 9단계 MQTT 옵션(`--mqtt-topic`)은 카메라 레포의 `common/mqtt_pub.py`와 `.mqtt.env`를 쓴다. 위치는 `CAM_LAB` 환경 변수(기본 `~/cam-lab`).
- YOLO·YOLOE·MobileCLIP 가중치는 ultralytics가 처음 실행할 때 내려받는다.

확인한 것: 정리된 구조에서 `tools/summarize_results.py` 출력이 원래 폴더 결과와 같고, `label_feat.py`가 만든 라벨이 저장소의 `data/`와 같으며, `eval_pose.py`가 그대로 돈다. 3~6단계 스크립트는 `common/eg2.py`와 `media/`를 찾는 것까지 확인했다.

올리지 않은 것: 모델 가중치(EmbeddingGemma 2 미세 조정본, YOLO·YOLOE·MobileCLIP), 학습 체크포인트, 데이터셋, 9단계 영상, 마이크 녹음.
