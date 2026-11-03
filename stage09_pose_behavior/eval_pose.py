"""9-6 처음 본 사람 시험: 학습에서 뺀 영상(--holdout)의 라벨 구간을 윈도로 잘라 모델 정확도·혼동 행렬을 낸다
라벨 구간은 label_feat.py의 ACTION/BEHAVIOR를 그대로 쓴다. 규칙 판정(CSV의 규칙 라벨 열)과도 비교한다
사용법: python eval_pose.py --model models/action_holdout.pt --videos 4492700 8189161 9019458 8502433 8165987
"""
import argparse, csv, glob, os, sys
import numpy as np
from pose_model import load_model, predict

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--videos", nargs="+", required=True, help="영상 번호 (앞 7자리)")
ap.add_argument("--stride", type=int, default=2)
args = ap.parse_args()

sys.argv = [sys.argv[0]]                 # label_feat.py의 인자 파싱이 이 스크립트 인자를 읽지 않게
src = open("label_feat.py", encoding="utf-8").read().split('print("[action]")')[0]   # 구간 정의만 읽고 파일 생성은 안 함
lf = {}
exec(compile(src, "label_feat.py", "exec"), lf)

ck = load_model(args.model)
seg_def = lf["ACTION"] if ck["task"] == "action" else lf["BEHAVIOR"]
keylen = 7 if ck["task"] == "action" else 8
RULE = {"만세": "both_up", "한팔들기": "one_up", "기본": "idle"}     # 규칙 라벨(프레임 단위) → 학습 라벨 이름
labels, win = ck["labels"], ck["window"]
cm = np.zeros((len(labels), len(labels)), int)
rule_hit = rule_n = 0
print(f"모델 {args.model} (window {win}, labels {labels})")
for path in sorted(glob.glob("runs/stage9/feat/*.csv")):
    key = os.path.basename(path)[:keylen]
    if key[:7] not in args.videos or key not in seg_def:
        continue
    rows = [r for r in csv.reader(open(path)) if r]
    t0 = float(rows[0][0])
    for label, a, b in seg_def[key]:
        if label not in labels:
            continue
        by_tid = {}
        for r in rows:
            if a <= float(r[0]) - t0 < b:
                by_tid.setdefault(r[1], []).append(r)
        n_ok = n = 0
        for rs in by_tid.values():
            feat = [np.array([float(x) for x in r[5:]] + [float(r[3]), float(r[4])], np.float32) for r in rs]
            for i in range(0, len(feat) - win + 1, args.stride):
                lab, p = predict(ck, feat[i:i + win])
                cm[labels.index(label), labels.index(lab)] += 1
                n += 1; n_ok += lab == label
                rl = max(set(r[2] for r in rs[i:i + win]), key=[r[2] for r in rs[i:i + win]].count)
                rule_n += 1; rule_hit += RULE.get(rl) == label
        print(f"  {key} {label:13s} 윈도 {n:3d}개  모델 정답 {n_ok}/{n}")
total = cm.sum()
print(f"\n처음 본 사람 정확도: {np.trace(cm)}/{total} = {np.trace(cm) / max(total, 1):.3f}"
      f"   (참고: 규칙 판정 다수결 정답 {rule_hit}/{rule_n}, 규칙에 없는 박수·팔짱·흔들기는 항상 오답)")
print("혼동 행렬 (행=정답, 열=예측)")
print(" " * 15 + "".join(f"{l[:8]:>9}" for l in labels))
for l, row in zip(labels, cm):
    print(f"{l:>14} " + "".join(f"{v:>9}" for v in row))
