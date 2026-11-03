"""10-2 카메라 위치 보정: 웹에서 두 카메라 스냅샷 위의 기준점을 클릭 → solvePnP로 카메라 위치·방향 계산
기준점 실제 좌표는 calib/world_points.json에 미리 적어 둔다 ("check": true 인 점은 계산에 쓰지 않고 검증에만 사용)
사용법: python calib_extrinsics.py --cams 0 1   →  브라우저에서 http://<Jetson IP>:8000
        python calib_extrinsics.py --cams rtsp://<맥북 IP>:8554/cam0 rtsp://<맥북 IP>:8554/cam1 --ids 0 1 --port 8005
"""
import argparse, json, os, time
import cv2, numpy as np, uvicorn
from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse, Response
from stereo import load_world, undistort, triangulate, load_cam, open_source, fit_res

ap = argparse.ArgumentParser()
ap.add_argument("--cams", nargs=2, default=["0", "1"], help="USB 웹캠 번호 또는 스트림 주소")
ap.add_argument("--ids", type=int, nargs=2, default=None, help="렌즈 보정 저장 번호 (calib/cam<id>_intr.npz). USB면 기본값 = 장치 번호")
ap.add_argument("--res", default="640x480")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--refine-f", action="store_true",
                help="기준점(8개 이상, 높이 차 0.3m 이상)으로 초점거리·k1도 추정 (체커보드 렌즈 보정이 불확실할 때)")
args = ap.parse_args()
W, H = map(int, args.res.split("x"))
if args.ids is None:
    assert all(c.isdigit() for c in args.cams), "스트림 주소를 쓸 때는 --ids 0 1 처럼 렌즈 보정 번호를 지정"
    args.ids = [int(c) for c in args.cams]
WORLD = load_world("calib/world_points.json")
WMAP = {p["id"]: p for p in WORLD}
CLICKS_F = "calib/clicks.json"
clicks = json.load(open(CLICKS_F)) if os.path.exists(CLICKS_F) else {"0": {}, "1": {}}
snaps = {}

def open_cam(i):
    cap = open_source(i, W, H)
    assert cap.isOpened(), f"카메라 {i}을 열 수 없음"
    return cap

def take_snaps():
    for n, ci in enumerate(args.cams):
        cap = open_cam(ci)
        for _ in range(15):          # 자동 노출 안정화
            ok, f = cap.read()
        cap.release()
        f = fit_res(f, W, H)
        snaps[str(n)] = f
        cv2.imwrite(f"calib/snap{n}.jpg", f)

def intr(n):
    d = np.load(f"calib/cam{args.ids[n]}_intr.npz")
    return d["K"], d["dist"]

def solve():
    out = {}
    for n in ("0", "1"):
        K, dist = intr(int(n))
        ids = [i for i in clicks[n] if i in WMAP and not WMAP[i].get("check")]
        if len(ids) < 6:
            out[n] = {"error": f"계산용 기준점 {len(ids)}개 (6개 이상 필요)"}; continue
        obj = np.float64([WMAP[i]["xyz"] for i in ids])
        img = np.float64([clicks[n][i] for i in ids])
        refined = ""
        if args.refine_f and len(ids) >= 8:
            # 렌즈 보정을 못 했거나 믿기 어려울 때: 높이가 서로 다른 기준점(바닥·책상·벽)으로 초점거리와 k1을 같이 추정
            # 주점(화면 중앙)·화소비는 고정. 기준점이 한 평면에만 있으면 초점거리가 정해지지 않는다
            if np.ptp(obj[:, 2]) < 0.3:
                refined = " (기준점 높이 차가 0.3m 미만이라 초점거리 추정 생략)"
            else:
                fl = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO |
                      cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
                f0 = K[0, 0]
                _, K, dist, _, _ = cv2.calibrateCamera([obj.astype(np.float32)], [img.astype(np.float32)], (W, H),
                                                       K.copy(), dist.copy(), flags=fl)
                np.savez(f"calib/cam{args.ids[int(n)]}_intr.npz", K=K, dist=dist, res=np.array([W, H]))
                refined = f" (초점거리 {f0:.0f} → {K[0, 0]:.0f}px, k1 {dist.ravel()[0]:.3f})"
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        errs = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
        R, _ = cv2.Rodrigues(rvec)
        np.savez(f"calib/cam{n}.npz", K=K, dist=dist, R=R, t=tvec, res=np.array([W, H]))
        C = (-R.T @ tvec).ravel()
        out[n] = {"pos": C.round(2).tolist(), "reproj": {i: round(float(e), 1) for i, e in zip(ids, errs)},
                  "mean_px": round(float(errs.mean()), 2), "f": round(float(K[0, 0]), 1) , "note": refined}
    # 두 카메라 모두 클릭한 점으로 3D 검증 (check 점 포함)
    if "error" not in out.get("0", {"error": 1}) and "error" not in out.get("1", {"error": 1}):
        c0, c1 = load_cam("calib/cam0.npz"), load_cam("calib/cam1.npz")
        ver = {}
        for i in set(clicks["0"]) & set(clicks["1"]) & set(WMAP):
            p0, p1 = undistort(c0, [clicks["0"][i]])[0], undistort(c1, [clicks["1"][i]])[0]
            X, _ = triangulate(c0, c1, p0, p1)
            ver[i] = {"check": bool(WMAP[i].get("check")),
                      "err_cm": round(float(np.linalg.norm(X - np.array(WMAP[i]["xyz"]))) * 100, 1)}
        out["verify"] = ver
    return out

app = FastAPI()

@app.get("/img/{n}.jpg")
def img(n: str):
    f = snaps[n].copy()
    for i, (u, v) in clicks[n].items():
        col = (0, 200, 255) if WMAP.get(i, {}).get("check") else (0, 255, 0)
        cv2.circle(f, (int(u), int(v)), 5, col, -1)
        cv2.putText(f, i, (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return Response(cv2.imencode(".jpg", f)[1].tobytes(), media_type="image/jpeg")

@app.post("/click")
def click(d: dict = Body(...)):
    clicks[str(d["cam"])][d["id"]] = [float(d["u"]), float(d["v"])]
    json.dump(clicks, open(CLICKS_F, "w"), indent=1)
    return {"ok": True}

@app.post("/undo")
def undo(d: dict = Body(...)):
    clicks[str(d["cam"])].pop(d["id"], None)
    json.dump(clicks, open(CLICKS_F, "w"), indent=1)
    return {"ok": True}

@app.post("/snap")
def snap():
    take_snaps(); return {"ok": True}

@app.post("/solve")
def do_solve():
    return solve()

@app.get("/points")
def points():
    return {"points": WORLD, "clicks": clicks}

PAGE = """<!doctype html><meta charset=utf-8><title>카메라 위치 보정</title>
<style>body{font-family:sans-serif;margin:16px;background:#111;color:#eee}.row{display:flex;gap:12px;flex-wrap:wrap}
.row div{flex:1;min-width:300px}img{width:100%;cursor:crosshair;border:1px solid #444}select,button{font-size:15px;padding:4px 10px}
pre{background:#222;padding:8px;white-space:pre-wrap}</style>
<h2>카메라 위치 보정</h2>
<p>기준점 선택 → 해당 카메라 이미지에서 그 점을 클릭. 초록 = 계산용, 주황 = 검증용(check). 카메라마다 계산용 6개 이상.</p>
<p><select id=pid></select> <button onclick="undo(0)">cam0 선택점 지우기</button> <button onclick="undo(1)">cam1 선택점 지우기</button>
<button onclick="snapNew()">새 스냅샷</button> <button onclick="solve()">계산</button></p>
<div class=row><div>cam0<br><img id=i0 src="/img/0.jpg" onclick="clk(event,0)"></div>
<div>cam1<br><img id=i1 src="/img/1.jpg" onclick="clk(event,1)"></div></div>
<h3>결과</h3><pre id=out>-</pre>
<script>
async function load(){const r=await (await fetch('/points')).json();
 document.getElementById('pid').innerHTML=r.points.map(p=>`<option value=${p.id}>${p.id} (${p.xyz.join(', ')})${p.check?' 검증':''}</option>`).join('');}
function refresh(){for(const n of [0,1])document.getElementById('i'+n).src=`/img/${n}.jpg?`+Date.now();}
async function clk(e,n){const im=e.target,u=e.offsetX*im.naturalWidth/im.clientWidth,v=e.offsetY*im.naturalHeight/im.clientHeight;
 await fetch('/click',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cam:n,id:pid.value,u,v})});refresh();}
async function undo(n){await fetch('/undo',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cam:n,id:pid.value})});refresh();}
async function snapNew(){await fetch('/snap',{method:'POST'});refresh();}
async function solve(){const r=await (await fetch('/solve',{method:'POST'})).json();document.getElementById('out').textContent=JSON.stringify(r,null,1);}
load();
</script>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE

if __name__ == "__main__":
    take_snaps()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
