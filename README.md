# EmbeddingGemma 2 멀티모달 임베딩 실습 (Jetson Orin Nano Super)

AIGJ_01_014 · team2 보드

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

## 실행 참고

보드에서는 모든 파일을 한 폴더(`~/eg2-lab`)에 두고 그 폴더에서 실행했다. 이 저장소는 보기 쉽게 단계별로 나눠 둔 것이라, 다시 실행하려면 `common/`과 해당 단계 폴더의 파일을 한 폴더에 모아 실행한다(스크립트 안 경로 `media/`, `data/`, `runs/` 기준).

올리지 않은 것: 모델 가중치(EmbeddingGemma 2 미세 조정본, YOLO·YOLOE·MobileCLIP), 데이터셋, 원본 영상·사진(사람이 나옴). 보고서의 영상 출처는 Pexels 공개 영상.
