"""10단계: USB 카메라 2대로 강의실 특정 물체(타겟)에 대한 사람의 접근 감지·알람
--mode ankle1 : 카메라 1대 + 발목 → 바닥(z=0) 위치 (10-3 문제 확인)
--mode stereo : 카메라 2대 + 상체(어깨 중심) 삼각측량 (10-4 개선)
--mode fused  : stereo + 한 카메라에만 보이는 사람은 높이 평면으로 추정 (10-5 보완, 기본)
타겟 위치: --target "문구"로 YOLOE 후보 + EmbeddingGemma 2 재확인 후 삼각측량, 또는 --target-xy x,y 직접 지정
--moving      : 타겟도 매 프레임 추적 (칼만 필터 + 상대 속도 + 충돌 예상 시간 TTC 알람) (10-8)
사용법: python proximity_app.py --cams 0 2 --target "red fire extinguisher"
        python proximity_app.py --cams 0 2 --target "cart" --moving
        python proximity_app.py --cams rtsp://<맥북 IP>:8554/cam0 rtsp://<맥북 IP>:8554/cam1 --target-xy 3.0,1.0 --port 8005
"""
import argparse, time, threading, collections
import cv2, numpy as np, uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO
import stereo as S

ap = argparse.ArgumentParser()
ap.add_argument("--cams", nargs=2, default=["0", "1"], help="USB 웹캠 번호 또는 스트림 주소. calib_extrinsics.py와 같은 순서")
ap.add_argument("--mode", choices=["ankle1", "stereo", "fused"], default="fused")
ap.add_argument("--target", default=None, help='예: "red fire extinguisher"')
ap.add_argument("--target-xy", default=None, help="예: 3.0,1.2 (직접 측정한 바닥 좌표)")
ap.add_argument("--target-h", type=float, default=0.8, help="한 카메라만 타겟을 볼 때 가정할 높이(m)")
ap.add_argument("--near", type=float, default=1.0, help="근접 알람 거리(m)")
ap.add_argument("--far", type=float, default=3.0, help="이탈 알람 거리(m)")
ap.add_argument("--moving", action="store_true", help="이동 타겟 모드 (10-8)")
ap.add_argument("--ttc", type=float, default=2.0, help="충돌 위험 알람 TTC 기준(초)")
ap.add_argument("--target-every", type=int, default=2, help="이동 타겟 검출 주기(프레임)")
ap.add_argument("--res", default="640x480")
ap.add_argument("--port", type=int, default=8000)
args = ap.parse_args()
W, H = map(int, args.res.split("x"))
if args.moving and not args.target:
    ap.error("--moving 에는 --target \"문구\"가 필요합니다 (매 프레임 YOLOE로 타겟을 찾기 때문)")

MAX_SKEW = 0.08       # 두 카메라 프레임 시간 차 허용(초). 넘으면 그 순간은 삼각측량 생략
H_GUESS = 1.2         # 한 카메라 추정 시 어깨 높이 가정(m): 앉음 약 1.0, 섬 약 1.4의 중간
GATE = 0.8            # 같은 사람으로 이어 붙일 최대 이동(m/프레임)
SMOOTH_S = 0.5        # 위치 중앙값 창(초)
SIT_H = 1.15          # 이 높이 미만이면 '앉음'
NUM_CAMS = 1 if args.mode == "ankle1" else 2
CLOSE_V = 0.5         # 충돌 위험 알람: 접근 속도(m/s) 기준
TGT_LOST_S = 2.0      # 이 시간 이상 타겟을 못 보면 EmbeddingGemma로 다시 식별

cams = [S.load_cam(f"calib/cam{n}.npz") for n in range(NUM_CAMS)]
ROOM = S.room_bounds(S.load_world("calib/world_points.json"))
det = YOLO("yolo11n-pose.pt")

# ---------------- 카메라 수신 (최신 프레임만 보관) ----------------
class Cam(threading.Thread):
    def __init__(self, dev):
        super().__init__(daemon=True)
        self.cap = S.open_source(dev, W, H)
        assert self.cap.isOpened(), f"카메라 {dev}을 열 수 없음 (Q11, Q14 참고)"
        self.lock, self.frame, self.ts = threading.Lock(), None, 0.0
    def run(self):
        while True:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame, self.ts = S.fit_res(f, W, H), time.time()
    def get(self):
        with self.lock:
            return self.ts, (None if self.frame is None else self.frame.copy())

# ---------------- 접근·이탈 알람 ----------------
class ProximityAlarm:
    """히스테리시스로 경계에서 깜빡이지 않게 하고, 최근 1초 거리 변화 속도로 접근/이탈 판단"""
    def __init__(self, near, far, margin=0.2, v_th=0.3):
        self.near, self.far, self.m, self.v_th = near, far, margin, v_th
        self.zone, self.hist, self.trend, self.ttc_on = "mid", [], "정지", False
    def update_ttc(self, closing, ttc, d):
        """이동 타겟: 빠르게 가까워지고 TTC가 짧으면 충돌 위험 알람 (조건이 풀리면 다시 무장)"""
        danger = closing > CLOSE_V and ttc is not None and ttc < args.ttc
        ev = []
        if danger and not self.ttc_on:
            ev.append(f"알람: 충돌 위험 TTC {ttc:.1f}초 (접근 {closing:.1f}m/s, {d:.1f}m)")
        if danger:
            self.ttc_on = True
        elif self.ttc_on and (closing < 0.2 or ttc is None or ttc > args.ttc * 1.5):
            self.ttc_on = False
        return ev
    def update(self, t, d):
        self.hist = [(tt, dd) for tt, dd in self.hist if t - tt < 1.0] + [(t, d)]
        span = self.hist[-1][0] - self.hist[0][0]
        v = (self.hist[-1][1] - self.hist[0][1]) / span if span > 0.3 else 0.0
        trend = "접근 중" if v < -self.v_th else "멀어지는 중" if v > self.v_th else "정지"
        ev = []
        if trend == "접근 중" and self.trend != "접근 중" and d < self.far:
            ev.append(f"접근 시작 ({d:.1f}m)")
        self.trend = trend
        if self.zone != "near" and d < self.near:
            self.zone = "near"; ev.append(f"알람: 근접 {d:.1f}m")
        elif self.zone == "near" and d > self.near + self.m:
            self.zone = "mid"
        if self.zone != "far" and d > self.far:
            if self.zone == "mid" and self.hist[0][1] <= self.far:   # 안쪽에 있다가 나간 경우만 알람
                ev.append(f"알람: 이탈 {d:.1f}m")
            self.zone = "far"
        elif self.zone == "far" and d < self.far - self.m:
            self.zone = "mid"
        return ev

# ---------------- 바닥 평면 추적 ----------------
class Track:
    _next = 1
    def __init__(self, xy, h, kind, now):
        self.id, Track._next = Track._next, Track._next + 1
        self.hist = collections.deque([(now, xy, h)], maxlen=60)
        self.kind, self.last, self.first = kind, now, now
        self.alarm = ProximityAlarm(args.near, args.far)
        self.kf = S.Kalman2D(xy)
        self.rel = (None, 0.0, None)          # (거리, 접근 속도, TTC)
    def observe(self, xy, h, kind, now):
        self.kf.predict(now - self.last); self.kf.update(xy)
        self.hist.append((now, xy, h)); self.kind, self.last = kind, now
    @property
    def xy(self):
        if args.moving:                       # 이동 모드: 칼만 필터 위치 (속도 추정 포함)
            return self.kf.x[:2].copy()
        pts = [p for t, p, _ in self.hist if self.last - t <= SMOOTH_S]
        return np.median(pts, axis=0)
    @property
    def h(self):
        return float(np.median([hh for t, _, hh in self.hist if self.last - t <= SMOOTH_S]))

tracks, events = [], collections.deque(maxlen=60)
ev_lock = threading.Lock()
state = {"target": None, "stats": {}, "people": []}
tgt = {"kf": None, "h": args.target_h, "last": 0.0, "boxes": [None, None], "how": "",
       "trail": collections.deque(maxlen=50), "ref": None}
models = {}           # 이동 모드에서 계속 쓰는 yoloe / emb / qv
jpegs = [None, None]

def log(now, who, msg):
    with ev_lock:
        events.appendleft({"ts": now, "who": who, "label": msg})

def update_tracks(obs, now):
    """obs: [(xy, h, kind)] → 가장 가까운 트랙에 이어 붙이고(GATE 이내), 나머지는 새 트랙"""
    free = list(range(len(tracks)))
    for xy, h, kind in sorted(obs, key=lambda o: o[2] != "stereo"):     # 삼각측량 결과 우선
        best, bi = GATE, None
        for i in free:
            d = np.linalg.norm(tracks[i].xy - xy)
            if d < best:
                best, bi = d, i
        if bi is None:
            tracks.append(Track(xy, h, kind, now)); log(now, f"#{tracks[-1].id}", "등장")
        else:
            tr = tracks[bi]; tr.observe(xy, h, kind, now)
            free.remove(bi)
    for tr in [t for t in tracks if now - t.last > 2.0]:
        tracks.remove(tr); log(now, f"#{tr.id}", f"퇴장 (머문 시간 {tr.last - tr.first:.0f}초)")

# ---------------- 타겟 찾기: YOLOE 후보 → EmbeddingGemma 2 재확인 → 삼각측량 ----------------
def load_target_models():
    import torch
    from ultralytics import YOLOE
    from sentence_transformers import SentenceTransformer
    if not models:
        yoloe = YOLOE("yoloe-11s-seg.pt")
        yoloe.set_classes([args.target], yoloe.get_text_pe([args.target]))
        emb = SentenceTransformer("google/embeddinggemma-2", device="cuda",
                                  model_kwargs={"torch_dtype": torch.bfloat16}, config_kwargs={"audio_config": None})
        qv = np.asarray(emb.encode(f"a photo of a {args.target}", prompt_name="SearchQuery",
                                   normalize_embeddings=True), dtype=np.float32)
        models.update(yoloe=yoloe, emb=emb, qv=qv)
    return models["yoloe"], models["emb"], models["qv"]

def embed_crop(emb, f, box, path):
    x1, y1, x2, y2 = box
    cv2.imwrite(path, f[y1:y2, x1:x2])
    return np.asarray(emb.encode({"image": path}, normalize_embeddings=True), dtype=np.float32)

def find_target(frames):
    import torch
    yoloe, emb, qv = load_target_models()
    found = []
    for n, f in enumerate(frames):
        r = yoloe.predict(f, conf=0.1, verbose=False)[0]
        best = None
        for i, (x1, y1, x2, y2) in enumerate(r.boxes.xyxy.int().tolist()):
            if min(x2 - x1, y2 - y1) < 16:
                continue
            v = embed_crop(emb, f, (x1, y1, x2, y2), f"calib/target_cand{n}_{i}.jpg")
            score = float(v @ qv)
            if best is None or score > best[0]:
                best = (score, ((x1 + x2) / 2, (y1 + y2) / 2), (x1, y1, x2, y2), v)
        found.append(best)
    ok = [b is not None and b[0] >= 0.15 for b in found]
    if any(ok):                                    # 이동 모드 재식별용 기준 임베딩 (점수가 높은 쪽 크롭)
        tgt["ref"] = max((b for b, o in zip(found, ok) if o), key=lambda b: b[0])[3]
    if not args.moving:
        models.clear()
        torch.cuda.empty_cache()                   # 고정 타겟은 한 번 찾고 메모리 반환
    if NUM_CAMS == 2 and all(ok):
        p0, p1 = S.undistort(cams[0], [found[0][1]])[0], S.undistort(cams[1], [found[1][1]])[0]
        X, err = S.triangulate(cams[0], cams[1], p0, p1)
        how = f"삼각측량 (재투영 {err:.1f}px)"
    elif any(ok):
        n = ok.index(True)
        X = S.ray_plane(cams[n], S.undistort(cams[n], [found[n][1]])[0], args.target_h)
        how = f"cam{n} 단독 추정 (높이 {args.target_h}m 가정)"
    else:
        return None
    return {"xyz": X.round(2).tolist(), "how": how,
            "boxes": [b[2] if o else None for b, o in zip(found, ok)],
            "scores": [round(b[0], 3) if b else None for b in found]}

def set_target(frames):
    if args.target_xy:
        x, y = map(float, args.target_xy.split(","))
        state["target"] = {"xyz": [x, y, 0.0], "how": "직접 지정", "boxes": [None, None], "scores": []}
    elif args.target:
        t = find_target(frames)
        state["target"] = t
        log(time.time(), "타겟", f"찾음: {t['xyz']} / {t['how']} / 점수 {t['scores']}" if t else "찾지 못함")
    if args.moving and state["target"]:
        x, y, z = state["target"]["xyz"]
        tgt.update(kf=S.Kalman2D((x, y)), last=time.time(), boxes=state["target"]["boxes"],
                   how=state["target"]["how"])
        if z > 0.1:
            tgt["h"] = z
        tgt["trail"].clear()
    retarget.clear()

def track_target(frames, now, idx):
    """10-8: 이동 타겟을 매 N프레임 검출 → 예측 위치에 가까운 후보 선택(놓쳤으면 임베딩으로 재식별) → 삼각측량 → 칼만 갱신"""
    kf = tgt["kf"]
    if kf is None:
        return
    if idx % args.target_every == 0:
        yoloe, emb, qv = load_target_models()
        lost = now - tgt["last"] > TGT_LOST_S
        X_pred = [*kf.at(now - tgt["last"]), tgt["h"]]
        chosen = [None, None]
        for n, f in enumerate(frames):
            r = yoloe.predict(f, conf=0.1, verbose=False)[0]
            boxes = [b for b in r.boxes.xyxy.int().tolist() if min(b[2] - b[0], b[3] - b[1]) >= 16]
            if not boxes:
                continue
            centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
            if not lost:                       # 추적 중: 예측 위치를 화면에 투영해 가장 가까운 후보 (100px 이내)
                pp = S.project_distorted(cams[n], X_pred)
                dists = [np.hypot(c[0] - pp[0], c[1] - pp[1]) for c in centers]
                i = int(np.argmin(dists))
                if dists[i] < 100:
                    chosen[n] = (centers[i], boxes[i])
            else:                              # 놓침: 기준 임베딩과 가장 닮은 후보 (큰 박스 5개까지)
                ref = tgt["ref"] if tgt["ref"] is not None else qv
                cand = sorted(range(len(boxes)), key=lambda i: -(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))[:5]
                sc = [(float(embed_crop(emb, f, boxes[i], f"calib/reid{n}_{i}.jpg") @ ref), i) for i in cand]
                s_best, i = max(sc)
                if s_best >= 0.15:
                    chosen[n] = (centers[i], boxes[i])
        X, how = None, ""
        if NUM_CAMS == 2 and all(chosen):
            p0, p1 = S.undistort(cams[0], [chosen[0][0]])[0], S.undistort(cams[1], [chosen[1][0]])[0]
            Xt, err = S.triangulate(cams[0], cams[1], p0, p1)
            if err < 15:
                X, how = Xt, "삼각측량"
                tgt["h"] = 0.8 * tgt["h"] + 0.2 * float(Xt[2])     # 타겟 높이는 천천히 갱신
        if X is None and any(chosen):
            n = 0 if chosen[0] else 1
            X = S.ray_plane(cams[n], S.undistort(cams[n], [chosen[n][0]])[0], tgt["h"])
            how = f"cam{n} 단독 추정"
        tgt["boxes"] = [c[1] if c else None for c in chosen]
        if X is not None:
            kf.predict(now - tgt["last"]); kf.update(X[:2])
            tgt["last"], tgt["how"] = now, how
            tgt["trail"].append([round(float(X[0]), 2), round(float(X[1]), 2)])
            if lost:
                log(now, "타겟", "다시 찾음 (임베딩 재식별)")
    pos = kf.at(now - tgt["last"])
    state["target"] = {"xyz": [round(float(pos[0]), 2), round(float(pos[1]), 2), round(tgt["h"], 2)],
                       "v": [round(float(kf.x[2]), 2), round(float(kf.x[3]), 2)],
                       "how": tgt["how"] + (" (예측)" if now - tgt["last"] > 0.3 else ""),
                       "lost": now - tgt["last"] > TGT_LOST_S,
                       "boxes": tgt["boxes"], "trail": list(tgt["trail"]), "scores": []}

# ---------------- 메인 처리 ----------------
def draw(n, res):
    vis = res.plot(labels=False, boxes=False)
    t = state["target"]
    box = t["boxes"][n] if t and n < len(t["boxes"]) else None
    if box:
        x1, y1, x2, y2 = box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(vis, "TARGET", (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    for tr in tracks:
        if now_global - tr.last > 0.3:
            continue
        u, v = S.project_distorted(cams[n], [*tr.xy, tr.h if args.mode != "ankle1" else 0.0])
        if 0 <= u < W and 0 <= v < H:
            col = (0, 0, 255) if tr.alarm.zone == "near" else (0, 255, 0)
            cv2.circle(vis, (int(u), int(v)), 6, col, -1)
            cv2.putText(vis, f"#{tr.id}", (int(u) + 8, int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.putText(vis, f"cam{n} | {args.mode}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()

retarget = threading.Event(); retarget.set()
now_global = 0.0

def process_loop():
    global now_global
    srcs = [Cam(d) for d in args.cams[:NUM_CAMS]]
    for s in srcs:
        s.start()
    t0, n_frames, idx = time.time(), 0, 0
    while True:
        got = [s.get() for s in srcs]
        if any(f is None for _, f in got):
            time.sleep(0.05); continue
        frames = [f for _, f in got]
        if retarget.is_set():
            set_target(frames)
        skew = abs(got[0][0] - got[-1][0])
        results = det.predict(frames, conf=0.5, half=True, verbose=False)
        kps = [np.concatenate([r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()[..., None]], -1)
               if r.keypoints is not None and r.keypoints.conf is not None else np.zeros((0, 17, 3)) for r in results]
        now = now_global = time.time()
        idx += 1
        if args.moving:
            track_target(frames, now, idx)
        obs = []
        if args.mode == "ankle1":
            for k in kps[0]:
                a = S.ankle(k)
                X = S.ray_plane(cams[0], S.undistort(cams[0], [a])[0], 0.0) if a is not None else None
                if X is not None:
                    obs.append((X[:2], 0.0, "ankle"))
        else:
            pts = [[(i, S.upper_body(k)) for i, k in enumerate(kk)] for kk in kps]
            pts = [[(i, p) for i, p in pp if p is not None] for pp in pts]
            und = [S.undistort(cams[n], [p for _, p in pts[n]]) for n in range(2)]
            matched, rest0, rest1 = ([], set(range(len(und[0]))), set(range(len(und[1]))))
            if skew <= MAX_SKEW:
                matched, rest0, rest1 = S.match_people(cams[0], cams[1], und[0], und[1])
            for i, j, X, err in matched:
                obs.append((X[:2], float(X[2]), "stereo"))
            if args.mode == "fused":
                singles = []
                for n, rest in ((0, rest0), (1, rest1)):
                    for i in rest:
                        X = S.ray_plane(cams[n], und[n][i], H_GUESS)
                        if X is not None:
                            singles.append(X[:2])
                for xy in singles:   # 삼각측량 결과나 이미 추가한 추정과 겹치면 같은 사람으로 보고 버림
                    if all(np.linalg.norm(xy - o[0]) > 0.6 for o in obs):
                        obs.append((xy, H_GUESS, "single"))
        x0, y0, x1, y1 = ROOM
        obs = [o for o in obs if x0 <= o[0][0] <= x1 and y0 <= o[0][1] <= y1]
        update_tracks(obs, now)

        people = []
        txy = np.array(state["target"]["xyz"][:2]) if state["target"] else None
        for tr in tracks:
            d = float(np.linalg.norm(tr.xy - txy)) if txy is not None else None
            closing, ttc = 0.0, None
            if d is not None:
                msgs = tr.alarm.update(now, d)
                if args.moving and tgt["kf"] is not None and not state["target"].get("lost"):
                    t_state = np.concatenate([txy, tgt["kf"].x[2:]])
                    p_state = np.concatenate([tr.kf.at(now - tr.last), tr.kf.x[2:]])
                    d, closing, ttc = S.relative_state(p_state, t_state)
                    msgs += tr.alarm.update_ttc(closing, ttc, d)
                for msg in msgs:
                    log(now, f"#{tr.id}", msg)
            people.append({"id": tr.id, "x": round(float(tr.xy[0]), 2), "y": round(float(tr.xy[1]), 2),
                           "h": round(tr.h, 2), "kind": tr.kind,
                           "pose": "-" if tr.kind != "stereo" else ("앉음" if tr.h < SIT_H else "섬"),
                           "d": None if d is None else round(d, 2), "trend": tr.alarm.trend, "zone": tr.alarm.zone,
                           "v": [round(float(tr.kf.x[2]), 2), round(float(tr.kf.x[3]), 2)],
                           "closing": round(closing, 2), "ttc": None if ttc is None else round(ttc, 1),
                           "danger": tr.alarm.ttc_on})
        n_frames += 1
        if now - t0 >= 1:
            state["stats"] = {"fps": round(n_frames / (now - t0), 1), "skew_ms": round(skew * 1000),
                              "mode": args.mode + (" +moving" if args.moving else "")}
            t0, n_frames = now, 0
        state["people"] = people
        for n in range(NUM_CAMS):
            jpegs[n] = draw(n, results[n])

app = FastAPI()

@app.get("/stream/{n}")
def stream(n: int):
    def gen():
        while True:
            f = jpegs[n] if n < len(jpegs) else None
            if f:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
            time.sleep(0.07)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/state")
def get_state():
    with ev_lock:
        ev = list(events)[:25]
    return {**state, "room": ROOM, "near": args.near, "far": args.far, "events": ev, "cams": NUM_CAMS,
            "moving": args.moving,
            "cam_pos": [c["C"].round(2).tolist() for c in cams]}

@app.post("/retarget")
def do_retarget():
    retarget.set(); return {"ok": True}

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>강의실 접근 감지</title>
<style>body{font-family:sans-serif;margin:16px;background:#111;color:#eee}.row{display:flex;gap:10px;flex-wrap:wrap}
.row>div{flex:1;min-width:300px}img{width:100%;border-radius:6px}svg{background:#1b1b1b;border-radius:6px;width:100%;max-width:640px}
table{border-collapse:collapse}td,th{border:1px solid #444;padding:3px 8px;font-size:13px}button{font-size:14px;padding:4px 10px}
.near{color:#ff5555;font-weight:bold}</style>
<h2>강의실 접근 감지 <span id=st></span></h2>
<p><button onclick="fetch('/retarget',{method:'POST'})">타겟 다시 찾기</button>
<button onclick="snd=true;beep()">알람 소리 켜기</button> <span id=tg></span></p>
<div class=row><div><img src="/stream/0"></div><div id=c1><img src="/stream/1"></div></div>
<div class=row><div><h3>평면도</h3><svg id=map viewBox="0 0 640 480"></svg></div>
<div><h3>사람</h3><table><thead><tr><th>ID</th><th>거리(m)</th><th>추세</th><th>구역</th><th>접근(m/s)</th><th>TTC(s)</th><th>자세</th><th>방식</th></tr></thead><tbody id=pp></tbody></table>
<h3>기록</h3><table><thead><tr><th>시각</th><th>대상</th><th>내용</th></tr></thead><tbody id=ev></tbody></table></div></div>
<script>
let snd=false,lastEv=0;
function beep(){try{const a=new AudioContext(),o=a.createOscillator();o.frequency.value=880;o.connect(a.destination);o.start();o.stop(a.currentTime+0.3);}catch(e){}}
async function poll(){const r=await (await fetch('/state')).json();
 const [x0,y0,x1,y1]=r.room,s=Math.min(600/(x1-x0),440/(y1-y0)),X=x=>20+(x-x0)*s,Y=y=>20+(y-y0)*s;
 const arrow=(x,y,v,c)=>r.moving&&Math.hypot(...v)>0.15?`<line x1=${X(x)} y1=${Y(y)} x2=${X(x+v[0])} y2=${Y(y+v[1])} stroke=${c} stroke-width=2 />`:'';
 if(r.cams<2)document.getElementById('c1').style.display='none';
 document.getElementById('st').textContent=r.stats.fps?`[${r.stats.mode}] ${r.stats.fps} fps, 카메라 시간 차 ${r.stats.skew_ms} ms`:'';
 let g=`<rect x=${X(x0)} y=${Y(y0)} width=${(x1-x0)*s} height=${(y1-y0)*s} fill=none stroke=#555 />`;
 r.cam_pos.forEach((c,i)=>g+=`<rect x=${X(c[0])-6} y=${Y(c[1])-6} width=12 height=12 fill=#4af /><text x=${X(c[0])+8} y=${Y(c[1])} fill=#4af font-size=12>cam${i}</text>`);
 const t=r.target;
 if(t){const tx=X(t.xyz[0]),ty=Y(t.xyz[1]);
  g+=`<circle cx=${tx} cy=${ty} r=${r.far*s} fill=none stroke=#666 stroke-dasharray=4 /><circle cx=${tx} cy=${ty} r=${r.near*s} fill=#ff000022 stroke=#f55 />`+
     `<rect x=${tx-7} y=${ty-7} width=14 height=14 fill=#f0f /><text x=${tx+10} y=${ty-8} fill=#f0f font-size=12>TARGET</text>`;
  if(t.trail&&t.trail.length>1)g+=`<polyline points="${t.trail.map(p=>X(p[0])+','+Y(p[1])).join(' ')}" fill=none stroke=#f0f stroke-opacity=0.5 />`;
  if(t.v)g+=arrow(t.xyz[0],t.xyz[1],t.v,'#f0f');
  document.getElementById('tg').textContent=`타겟 ${t.xyz.join(', ')} (${t.how})`+(t.v?` 속도 ${Math.hypot(...t.v).toFixed(2)}m/s`:'')+(t.lost?' — 놓침':'');}
 else document.getElementById('tg').textContent='타겟 없음';
 r.people.forEach(p=>{const c=(p.zone=='near'||p.danger)?'#f55':p.kind=='single'?'#aa0':'#5f5';
  g+=arrow(p.x,p.y,p.v,c);
  g+=`<circle cx=${X(p.x)} cy=${Y(p.y)} r=8 fill=${c} ${p.kind=='single'?'fill-opacity=0.5':''} /><text x=${X(p.x)+10} y=${Y(p.y)+4} fill=${c} font-size=13>#${p.id} ${p.d??''}</text>`;});
 document.getElementById('map').innerHTML=g;
 document.getElementById('pp').innerHTML=r.people.map(p=>`<tr class=${(p.zone=='near'||p.danger)?'near':''}><td>#${p.id}</td><td>${p.d??'-'}</td><td>${p.trend}</td><td>${p.zone}</td><td>${p.closing}</td><td>${p.ttc??'-'}</td><td>${p.pose}</td><td>${p.kind}</td></tr>`).join('');
 document.getElementById('ev').innerHTML=r.events.map(e=>`<tr><td>${new Date(e.ts*1000).toLocaleTimeString()}</td><td>${e.who}</td><td>${e.label}</td></tr>`).join('');
 const fresh=r.events.filter(e=>e.ts>lastEv&&e.label.startsWith('알람'));if(fresh.length&&snd)beep();
 if(r.events.length)lastEv=Math.max(lastEv,r.events[0].ts);}
setInterval(poll,500);poll();
</script>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE

if __name__ == "__main__":
    threading.Thread(target=process_loop, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
