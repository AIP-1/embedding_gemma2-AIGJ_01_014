import os, time, threading, collections, argparse, glob
import cv2, numpy as np, torch, uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer

# ---------------- 실행 모드 ----------------
# yolo    : YOLO 클래스 그대로 사용 (문제 확인용 기준선)
# relabel : YOLO 박스 + EmbeddingGemma 2 재라벨 (개선 1)
# yoloe   : 텍스트로 클래스를 지정하는 YOLOE 검출기 (개선 2)
# ---------------- 입력 (--source) ----------------
# 0, 1 ...            : 웹캠 번호 (기본 0, 원래 동작)
# http://..., rtsp:// : 네트워크 스트림 (맥북·휴대폰 카메라 등)
# src                 : 폴더 안의 이미지·동영상을 이름 순서대로 한 번씩 분석한 뒤 결과 화면에서 멈춤 (--loop면 반복)
# src/clip.mp4        : 파일 하나만 분석
# 웹에서 파일을 업로드하거나 ▶ 버튼을 누르면 그 파일을 한 번 분석하고 결과 화면에서 멈춘다.
# (웹캠·스트림 입력일 때는 파일 분석 후 다시 카메라로 돌아간다)
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["yolo", "relabel", "yoloe"], default="relabel")
ap.add_argument("--conf", type=float, default=None, help="검출 신뢰도 (기본: 모드별 값)")
ap.add_argument("--port", type=int, default=8005)
ap.add_argument("--source", default="0", help="웹캠 번호 / 스트림 주소 / 이미지·동영상 폴더 / 파일 하나")
ap.add_argument("--hold", type=float, default=4.0, help="이미지 1장을 보여 줄 시간(초)")
ap.add_argument("--loop", action="store_true", help="폴더를 끝없이 반복 재생 (기본: 한 번씩 분석 후 멈춤)")
args = ap.parse_args()
MODE = args.mode

# ---------------- 설정 ----------------
W, H          = 1280, 720
CONF = args.conf if args.conf is not None else {"yolo": 0.5, "relabel": 0.35, "yoloe": 0.25}[MODE]
MIN_SIDE      = 48                    # 크롭 최소 크기(px)
REFRESH_S     = 30                    # 같은 추적 ID 재임베딩 간격(초)
SKIP_CLASSES  = {"person"}            # 크롭·저장하지 않을 클래스
DIM, MAX_ITEMS = 512, 1000
MODEL_ID      = "google/embeddinggemma-2"   # 7단계 모델은 텍스트 전용이라 여기서는 원본 사용
CROP_DIR      = f"crops/{MODE}"       # 모드별로 크롭을 따로 저장해 나중에 비교
SRC_DIR       = args.source if os.path.isdir(args.source) else "src"   # 업로드 파일 저장 폴더
IMG_EXT       = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
VID_EXT       = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
MAX_UPLOAD_MB = 500

# YOLO(COCO 80종)에 없는 물체는 비슷한 클래스로 오인식된다 (세탁기→oven, 전지 포장→cell phone).
# relabel: 크롭 임베딩과 아래 라벨 문구의 유사도로 라벨을 다시 붙인다.
# yoloe  : 아래 영문 라벨을 검출 클래스 자체로 사용한다.  (영문 = 영상 표시, 한글 = 웹 표시)
LABELS = [
    ("drum washing machine", "드럼 세탁기"), ("top-load washing machine", "통돌이 세탁기"),
    ("oven", "오븐"), ("microwave", "전자레인지"), ("refrigerator", "냉장고"),
    ("cell phone", "휴대폰"), ("battery pack", "원통형 전지 포장"), ("AA battery", "건전지"),
    ("laptop", "노트북"), ("monitor", "모니터"), ("keyboard", "키보드"), ("mouse", "마우스"),
    ("cup", "컵"), ("water bottle", "물병"), ("book", "책"), ("backpack", "가방"), ("chair", "의자"),
]
LABEL_MIN     = 0.20                  # 이 점수 미만이면 YOLO 라벨 유지 (웹의 점수를 보고 조정)
os.makedirs(CROP_DIR, exist_ok=True)
os.makedirs(SRC_DIR, exist_ok=True)

if MODE == "yoloe":
    from ultralytics import YOLOE
    names = [en for en, _ in LABELS]
    det = YOLOE("yoloe-11s-seg.pt")     # 첫 실행 시 자동 다운로드
    det.set_classes(names, det.get_text_pe(names))
else:
    det = YOLO("yolo11n.pt")          # 첫 실행 시 자동 다운로드
KO = dict(LABELS)                     # 영문 -> 한글
print(f"[mode={MODE}] conf={CONF} source={args.source} (업로드 폴더: {SRC_DIR})")
emb = SentenceTransformer(MODEL_ID, device="cuda",
        model_kwargs={"torch_dtype": torch.bfloat16},
        config_kwargs={"audio_config": None})   # 텍스트+비전(440M)

with torch.inference_mode():
    label_vecs = np.asarray(emb.encode([f"a photo of a {en}" for en, _ in LABELS],
        prompt_name="SearchQuery", truncate_dim=DIM, normalize_embeddings=True), dtype=np.float32)

frame_lock, emb_lock, idx_lock = threading.Lock(), threading.Lock(), threading.Lock()
latest_jpeg = None
crop_q = collections.deque(maxlen=64)
items = []          # {id, cls, conf, ts, img, vec}
seen = {}           # track_id -> 마지막 임베딩 시각
track_label = {}    # track_id -> (영문 라벨, 한글 라벨, 점수)
stats = {"fps": 0.0, "embed_ms": 0.0, "det_ms": 0.0}   # fps: 화면 처리 속도, det_ms: YOLO 1프레임 처리 시간(평균)
SRC = {"now": "", "req": None}       # 지금 재생 중인 입력 이름, 웹에서 재생을 요청한 파일 경로

# ---------------- 입력 소스 ----------------
def fit(img):
    """비율을 유지해 W x H 캔버스 가운데에 놓는다 (세로 사진·영상도 잘리지 않게)"""
    h, w = img.shape[:2]
    s = min(W / w, H / h)
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    canvas = np.zeros((H, W, 3), np.uint8)
    y0, x0 = (H - img.shape[0]) // 2, (W - img.shape[1]) // 2
    canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    return canvas

def idle_frame(msg):
    f = np.zeros((H, W, 3), np.uint8)
    cv2.putText(f, msg, (40, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    return f

def list_files():
    return sorted(p for p in glob.glob(os.path.join(SRC_DIR, "*")) if p.lower().endswith(IMG_EXT + VID_EXT))

def play_image(path):
    img = cv2.imread(path)
    if img is None:
        print("[source] 이미지 읽기 실패:", path); return
    img = fit(img)
    t_end = time.time() + args.hold
    while time.time() < t_end and SRC["req"] is None:
        yield img.copy()
        time.sleep(0.05)

def play_video(path):
    """실제 재생 속도에 맞춰 내보낸다. 처리가 느리면 중간 프레임을 건너뛴다 (웹캠과 같은 동작)"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("[source] 동영상 열기 실패 (H.264 mp4로 변환해 보세요):", path); return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    start, idx = time.time(), -1
    while SRC["req"] is None:
        t = time.time() - start
        target = int(t * fps)
        if target <= idx:                       # 처리가 빠르면 기다림
            time.sleep(max(0.0, (idx + 1) / fps - t)); continue
        while idx < target - 1 and cap.grab():  # 처리가 느리면 건너뜀
            idx += 1
        ok, f = cap.read()
        if not ok:
            break
        idx += 1
        yield fit(f)
    cap.release()

def play_file(path):
    SRC["now"] = os.path.basename(path)
    print("[source] 재생:", SRC["now"])
    yield from (play_video(path) if path.lower().endswith(VID_EXT) else play_image(path))

def take_request():
    p, SRC["req"] = SRC["req"], None
    return p

def live_source(src):
    """웹캠 번호 또는 스트림 주소. 별도 스레드가 최신 프레임만 보관해 지연이 쌓이지 않게 한다."""
    if src.isdigit():
        cap = cv2.VideoCapture(int(src), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    else:
        cap = cv2.VideoCapture(src)
    assert cap.isOpened(), f"입력을 열 수 없음: {src} (웹캠이면 Q11 참고, 다른 프로그램이 쓰고 있는지 확인)"
    latest = {"f": None}
    def reader():
        while True:
            ok, f = cap.read()
            if ok:
                latest["f"] = f
            else:
                time.sleep(0.05)
    threading.Thread(target=reader, daemon=True).start()
    label = f"camera {src}" if src.isdigit() else "stream"
    SRC["now"] = label
    while True:
        if SRC["req"]:
            yield from play_file(take_request())
            SRC["now"] = label
            continue
        f = latest["f"]
        if f is None:
            time.sleep(0.02); continue
        latest["f"] = None
        yield f if f.shape[:2] == (H, W) else fit(f)

def wait_request():
    """마지막 결과 화면을 그대로 두고 웹 요청을 기다린다.
    프레임을 내보내지 않으므로 YOLO도 쉬고, 같은 물체가 중복 저장되지도 않는다."""
    while SRC["req"] is None:
        time.sleep(0.1)

def file_source(src):
    single = src if os.path.isfile(src) else None
    files = [single] if single else list_files()
    if not files:
        SRC["now"] = "waiting for upload"
        yield idle_frame("No files - upload images/videos on the web page")
    while True:
        for p in files:
            if SRC["req"]:
                break
            yield from play_file(p)
        if args.loop and files and not SRC["req"]:
            files = [single] if single else list_files()
            continue
        if files and not SRC["req"]:
            SRC["now"] += " - done"
            print("[source] 분석 끝. 웹에서 업로드하거나 ▶ 버튼을 누르면 다음 파일을 분석합니다")
        wait_request()
        files = [take_request()]

def frame_source():
    src = args.source
    if src.isdigit() or "://" in src:
        yield from live_source(src)
    else:
        assert os.path.exists(src), f"입력 경로가 없음: {src}"
        yield from file_source(src)

# ---------------- 검출·임베딩 ----------------
def capture_loop():
    global latest_jpeg
    t0, n = time.time(), 0
    for frame in frame_source():
        t_det = time.perf_counter()
        r = det.track(frame, persist=True, conf=CONF, quantize=16, verbose=False)[0]
        d_ms = (time.perf_counter() - t_det) * 1000
        stats["det_ms"] = d_ms if stats["det_ms"] == 0 else 0.9 * stats["det_ms"] + 0.1 * d_ms
        now = time.time()
        if r.boxes.id is not None:
            for (x1, y1, x2, y2), tid, c, cf in zip(r.boxes.xyxy.int().tolist(),
                    r.boxes.id.int().tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                name = det.names[c]
                if name in SKIP_CLASSES or min(x2 - x1, y2 - y1) < MIN_SIDE:
                    continue
                if now - seen.get(tid, 0) < REFRESH_S:
                    continue
                seen[tid] = now
                crop_q.append((tid, name, cf, now, frame[y1:y2, x1:x2].copy()))
        vis = frame.copy()
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
        for (x1, y1, x2, y2), tid, c, cf in zip(r.boxes.xyxy.int().tolist(), ids,
                r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
            yolo = det.names[c]
            en = track_label[tid][0] if (MODE == "relabel" and tid in track_label) else yolo
            color = (0, 200, 255) if en != yolo else (0, 255, 0)   # 재라벨된 박스는 주황색
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            txt = f"{en} {cf:.2f}"                                  # yolo·yoloe: 검출기 라벨 + 검출 신뢰도
            if MODE == "relabel" and tid in track_label:            # relabel: 임베딩 라벨 + 재라벨 점수 | 원래 YOLO 판단
                txt = f"{en} {track_label[tid][2]:.2f} | yolo {yolo} {cf:.2f}"
            elif MODE == "yoloe" and tid in track_label:            # yoloe: 검출 신뢰도 | 임베딩 재확인 점수
                txt = f"{en} {cf:.2f} | emb {track_label[tid][2]:.2f}"
            cv2.putText(vis, txt, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        n += 1
        if now - t0 >= 1:
            stats["fps"], t0, n = n / (now - t0), now, 0
        cv2.putText(vis, f"{stats['fps']:.1f} fps | det {stats['det_ms']:.0f} ms | indexed {len(items)} | {SRC['now']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with frame_lock:
                latest_jpeg = buf.tobytes()

def embed_loop():
    seq = 0
    while True:
        if not crop_q:
            time.sleep(0.05); continue
        batch = [crop_q.popleft() for _ in range(min(8, len(crop_q)))]
        paths = []
        for tid, name, cf, ts, crop in batch:
            seq += 1
            p = f"{CROP_DIR}/{seq:06d}_{name}.jpg"
            cv2.imwrite(p, crop)
            paths.append(p)
        t = time.perf_counter()
        with emb_lock:
            vecs = emb.encode([{"image": p} for p in paths],
                              truncate_dim=DIM, normalize_embeddings=True)
        stats["embed_ms"] = (time.perf_counter() - t) * 1000 / len(paths)
        vecs = np.asarray(vecs, dtype=np.float32)
        sims = vecs @ label_vecs.T
        with idx_lock:
            for (tid, name, cf, ts, _), p, v, sim in zip(batch, paths, vecs, sims):
                j = int(sim.argmax())
                if MODE == "yoloe":              # yoloe: 검출기 라벨 그대로, 점수는 '그 라벨' 문구와의 임베딩 유사도 (재확인 점수)
                    k = next((i for i, (e, _) in enumerate(LABELS) if e == name), j)
                    en, ko, ls = name, KO.get(name, name), float(sim[k])
                elif MODE == "yolo":             # yolo: 검출기 라벨 그대로, 점수는 임베딩이 가장 비슷하다고 본 라벨의 유사도
                    en, ko, ls = name, KO.get(name, name), float(sim[j])
                elif sim[j] >= LABEL_MIN:
                    en, ko, ls = LABELS[j][0], LABELS[j][1], float(sim[j])
                else:
                    en, ko, ls = name, name, float(sim[j])
                track_label[tid] = (en, ko, ls)
                items.append({"id": tid, "cls": name, "label": ko, "lscore": round(ls, 3),
                              "conf": round(cf, 2), "ts": ts, "img": "/" + p, "vec": v})
            while len(items) > MAX_ITEMS:
                old = items.pop(0)
                try: os.remove(old["img"][1:])
                except OSError: pass

# ---------------- 웹 ----------------
app = FastAPI()
app.mount("/crops", StaticFiles(directory="crops"), name="crops")   # /crops/<mode>/<파일>

def public(it, score=None):
    d = {k: it[k] for k in ("id", "cls", "label", "lscore", "conf", "ts", "img")}
    if score is not None: d["score"] = round(float(score), 3)
    return d

@app.get("/stream")
def stream():
    def gen():
        while True:
            with frame_lock:
                f = latest_jpeg
            if f:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
            time.sleep(0.05)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/search")
def search(q: str, k: int = 12):
    with idx_lock:
        snap = list(items)
    if not snap:
        return []
    with emb_lock:
        qv = emb.encode(q, prompt_name="SearchQuery", truncate_dim=DIM, normalize_embeddings=True)
    s = np.stack([it["vec"] for it in snap]) @ np.asarray(qv, dtype=np.float32)
    return [public(snap[i], s[i]) for i in s.argsort()[::-1][:k]]

def mem_stats():
    """이 프로세스만의 메모리 (공용 보드에서 다른 사람 프로그램과 섞이지 않음)"""
    kb = {}
    for line in open("/proc/self/status"):
        if line.startswith(("VmRSS", "VmHWM")):
            k, v = line.split(":")
            kb[k] = int(v.split()[0])
    return {"rss_mb": kb.get("VmRSS", 0) // 1024,                       # 현재 일반 메모리
            "rss_peak_mb": kb.get("VmHWM", 0) // 1024,                  # 일반 메모리 최고값
            "gpu_mb": torch.cuda.memory_reserved() // 2**20,            # PyTorch가 잡은 GPU 메모리 (EmbeddingGemma 등)
            "gpu_peak_mb": torch.cuda.max_memory_reserved() // 2**20}

@app.get("/recent")
def recent(n: int = 12):
    with idx_lock:
        snap = items[-n:][::-1]
    st = dict(stats, det_fps_max=round(1000 / stats["det_ms"], 1) if stats["det_ms"] else 0.0, **mem_stats())
    return {"mode": MODE, "stats": st, "total": len(items), "source": SRC["now"],
            "items": [public(it) for it in snap]}

@app.post("/upload")
async def upload(request: Request, name: str, play: int = 1):
    """웹에서 파일 본문을 그대로 보내면 SRC_DIR에 저장 (python-multipart 없이 동작)"""
    base = os.path.basename(name).replace("/", "_")
    if not base.lower().endswith(IMG_EXT + VID_EXT):
        return {"ok": False, "error": "이미지·동영상 파일만 올릴 수 있음"}
    path = os.path.join(SRC_DIR, base)
    tmp, size = path + ".part", 0
    with open(tmp, "wb") as f:
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 2**20:
                f.close(); os.remove(tmp)
                return {"ok": False, "error": f"{MAX_UPLOAD_MB}MB 초과"}
            f.write(chunk)
    os.replace(tmp, path)
    if play:
        SRC["req"] = path
    return {"ok": True, "name": base, "mb": round(size / 2**20, 1)}

@app.get("/files")
def files():
    return {"now": SRC["now"], "files": [os.path.basename(p) for p in list_files()]}

@app.post("/play")
def play(name: str):
    path = os.path.join(SRC_DIR, os.path.basename(name))
    if not os.path.isfile(path):
        return {"ok": False, "error": "파일 없음"}
    SRC["req"] = path
    return {"ok": True}

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>Jetson 물체 검색</title>
<style>body{font-family:sans-serif;margin:16px;background:#111;color:#eee}
img.live{width:100%;max-width:960px;border-radius:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px}
.card{background:#222;padding:6px;border-radius:6px;font-size:12px}
.card img{width:100%;height:110px;object-fit:contain;background:#000}
input{font-size:16px;padding:6px;width:60%}button{font-size:16px;padding:6px 12px;margin:2px}
.files button{font-size:13px}</style>
<h2>실시간 영상 <span id=mode></span></h2><img class=live src="/stream">
<h2>파일 등록 <span id=now style="font-size:14px;color:#9c9"></span></h2>
<input type=file id=f multiple accept="image/*,video/*" style="width:auto">
<button onclick="up()">업로드 후 재생</button> <span id=upmsg></span>
<div id=files class=files style="margin-top:8px"></div>
<h2>텍스트로 물체 검색</h2>
<form onsubmit="go();return false"><input id=q placeholder="예: 빨간 컵, 열려 있는 노트북"><button>검색</button></form>
<div id=res class=grid></div>
<h2>최근 추출 <span id=st></span></h2><div id=rec class=grid></div>
<script>
const $=id=>document.getElementById(id);
const card=it=>`<div class=card><img src="${it.img}"><div><b>${it.label}</b> (${it.lscore})<br>검출기: ${it.cls} ${it.conf}`+
 (it.score!==undefined?`<br>검색 점수 <b>${it.score}</b>`:'')+`<br>${new Date(it.ts*1000).toLocaleTimeString()}</div></div>`;
async function go(){const q=$('q').value;if(!q)return;
 const r=await (await fetch('/search?q='+encodeURIComponent(q))).json();
 $('res').innerHTML=r.map(card).join('')||'결과 없음';}
async function up(){const fs=[...$('f').files];if(!fs.length){$('upmsg').textContent='파일을 고르세요';return;}
 for(const [i,f] of fs.entries()){$('upmsg').textContent=`업로드 중 ${f.name} (${(f.size/1048576).toFixed(1)}MB)`;
  const r=await (await fetch(`/upload?name=${encodeURIComponent(f.name)}&play=${i==0?1:0}`,{method:'POST',body:f})).json();
  $('upmsg').textContent=r.ok?`완료: ${r.name} (${r.mb}MB)`:`실패: ${r.error}`;}
 $('f').value='';files();}
async function play(n){await fetch('/play?name='+encodeURIComponent(n),{method:'POST'});}
async function files(){const r=await (await fetch('/files')).json();
 $('files').innerHTML=r.files.map(n=>`<button data-n="${encodeURIComponent(n)}" onclick="play(decodeURIComponent(this.dataset.n))">▶ ${n}</button>`).join('')||'등록된 파일 없음';}
async function poll(){const r=await (await fetch('/recent')).json();
 $('mode').textContent=`[mode: ${r.mode}]`;
 $('now').textContent=r.source?`재생 중: ${r.source}`:'';
 const s=r.stats;
 $('st').textContent=`(총 ${r.total}개, 처리 ${s.fps.toFixed(1)} fps, 검출 ${s.det_ms.toFixed(0)} ms = 최대 ${s.det_fps_max} fps, `+
  `임베딩 ${s.embed_ms.toFixed(0)} ms/개, RAM ${s.rss_mb}+${s.gpu_mb} MB (최고 ${s.rss_peak_mb}+${s.gpu_peak_mb}))`;
 $('rec').innerHTML=r.items.map(card).join('');}
setInterval(poll,3000);poll();files();
</script>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE

if __name__ == "__main__":
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=embed_loop, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
