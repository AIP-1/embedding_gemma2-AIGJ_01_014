"""9-3 대체: 웹캠 녹화 대신 동영상 분석(--record)으로 모은 관절 특징 CSV를 구간별로 잘라 라벨 파일로 만든다
영상 시작 = 그 CSV의 첫 행 시각으로 보고, 아래 구간(시작초, 끝초)의 행만 라벨 파일에 붙인다
결과: data/action/<라벨>.csv, data/behavior/<라벨>.csv
  → python train_pose.py --data data/action --task action --window 30 --out models/action.pt
사용법: python label_feat.py [--holdout 영상번호 ...]   (지정한 영상은 학습에서 빼고 시험용으로 남김)
"""
import argparse, csv, glob, os

ap = argparse.ArgumentParser()
ap.add_argument("--holdout", nargs="*", default=[], help="학습에서 뺄 영상 번호 (앞 7자리)")
args = ap.parse_args()

FEAT_DIR = "results/stage9/feat"
# 영상 이름 앞 7자리: [(라벨, 시작초, 끝초)]  (프레임 시트를 보고 정한 정답 구간)
ACTION = {
    "4586958": [("wave", 0, 13)],                               # 한 손 계속 흔들기
    "4492700": [("wave", 0, 3), ("idle", 6.5, 10.5)],           # 흔들기 → (손 키스) → 정면 정지
    "6668697": [("both_up", 0, 5), ("one_up", 6, 15)],          # 트로피 만세 → 한 팔만 들기
    "8502433": [("both_up", 0, 2.5)],                           # 측면, 앉아서 두 팔 머리 위
    "8165987": [("idle", 0, 3.5), ("one_up", 5, 12)],           # 앉아 정지 → 한 팔 들어 가리키기
    "8135560": [("idle", 2, 13.5)],                             # 두 사람 앉아 정면 대화
    "6194926": [("idle", 0, 12.7)],                             # 서서 정면 정지 (마스크)
    "7653227": [("idle", 0, 21.5)],                             # 서서 폰 보기
    "3255163": [("clap", 0, 7)],                                # 앉은 청중 박수
    "8189161": [("clap", 0, 7.7)],                              # 단체 박수
    "9019458": [("idle", 0, 1.5), ("arms_crossed", 2.8, 10)],   # 정면, 팔 X자 교차
    "3255223": [("arms_crossed", 0, 8)],                        # 팔짱 클로즈업
    "4586617": [("wave", 0, 13.4)],                             # 계속 손 흔들기 (양손 번갈아)
    "6800277": [("both_up", 0, 7)],                             # 뒷모습, 두 팔 머리 위
    "6944447": [("one_up", 0, 4.5)],                            # 한 팔 높이 들기
    "7202831": [("one_up", 5, 18.7)],                           # 4명 한 주먹 들기
    "7203594": [("one_up", 0, 18.2)],                           # 한 팔 든 채 유지
}
BEHAVIOR = {
    "13404996": [("passing", 0, 10.8)], "15713467": [("passing", 0, 12.7)],
    "15808538": [("passing", 0, 10.3)], "5319441-": [("passing", 0, 20.5)],
    "17386177": [("glance", 0, 3)],                             # 카메라 보며 다가옴
    "6194926-": [("interested", 0, 12.7)], "4492700-": [("interested", 0, 10.3)],
    "8135560-": [("interested", 0, 13.4)],
    "7653227-": [("linger", 0, 21.5)],                          # 폰 보며 서 있음
    "15007797": [("passing", 0, 30.9)],                         # 거리, 여러 명 지나감
    "3205917-": [("glance", 0, 7.5)],                           # 걸으며 카메라를 잠깐씩 봄
}

def build(segments, out_dir, keylen):
    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(f"{out_dir}/*.csv"):
        os.remove(old)
    out = {}
    for path in sorted(glob.glob(f"{FEAT_DIR}/*.csv")):
        key = os.path.basename(path)[:keylen]
        if key not in segments or key[:7] in args.holdout:
            continue
        rows = [r for r in csv.reader(open(path)) if r]
        if not rows:
            continue
        t0 = float(rows[0][0])
        for label, a, b in segments[key]:
            sel = [r[:1] + [f"{key}_{r[1]}"] + r[2:] for r in rows if a <= float(r[0]) - t0 < b]  # 추적 ID를 영상별로 구분
            out.setdefault(label, []).extend(sel)
            print(f"  {key} {label:13s} {a:>4}~{b:<4}s  {len(sel)}행")
    for label, rows in out.items():
        with open(f"{out_dir}/{label}.csv", "w", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"→ {out_dir}/{label}.csv  {len(rows)}행")

print("[action]");   build(ACTION, "data/action", 7)
print("[behavior]"); build(BEHAVIOR, "data/behavior", 8)
