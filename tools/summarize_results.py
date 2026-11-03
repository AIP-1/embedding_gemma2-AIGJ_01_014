"""runs/ 아래 측정 결과(6·7·8·9단계)를 마크다운 표로 정리해 출력
사용법: python summarize_results.py > runs/results_tables.md
"""
import glob, json, os, re

def tegra_gpu(path):
    """tegrastats 로그의 GR3D 사용률: 인코딩 반복 구간(상위 20개 샘플 = 약 10초) 평균"""
    try:
        v = [int(x) for x in re.findall(r"GR3D_FREQ (\d+)%", open(path).read())]
    except OSError:
        return None
    top = sorted(v)[-20:]
    return round(sum(top) / len(top)) if top else None

def measure(path):
    s = open(path).read()
    g = lambda pat: (re.search(pat, s) or [None, None])[1]
    return {"total": g(r"합계 \(근사\)\s*: ([\d,]+) MB"), "ms": g(r"1회 평균 (\d+) ms"),
            "n": g(r"10초 동안 (\d+)회")}

print("## 6단계 구성별 자원 사용량\n")
print("| 구성 | 로드 크기 | 로드 후 RAM (MB, 이 프로세스 RSS+GPU) | 입력 | 지연 (ms, 1회 평균) | GPU 사용률 (%) |")
print("| --- | --- | --- | --- | --- | --- |")
rows = [("text", "텍스트 전용", "270M", "질의 2개"), ("image", "+비전", "440M", "이미지 5장"),
        ("video", "+비전", "440M", "동영상 10초"), ("audio", "+오디오", "570M", "WAV 3개"),
        ("all", "전체", "740M", "혼합")]
for key, name, size, inp in rows:
    p = f"runs/stage6/{key}.log"
    if not os.path.exists(p):
        continue
    m = measure(p)
    print(f"| {name} | {size} | {m['total']} | {inp} | {m['ms']} | {tegra_gpu(f'runs/stage6/{key}.tegra')} |")

print("\n## 7단계 학습 전후 비교\n")
print("| 모델 | accuracy@1 | accuracy@3 | ndcg@10 | 학습 시간 |")
print("| --- | --- | --- | --- | --- |")
for key, name, tt in [("base", "기준 (google/embeddinggemma-2)", "-"),
                      ("finetuned", "미세 조정 (eg2-classroom)", "228 s (3 epoch, 54 step)")]:
    p = f"runs/stage7/{key}.log"
    if os.path.exists(p):
        s = open(p).read()
        g = lambda k: (re.search(k + r" ([\d.]+)", s) or [None, "?"])[1]
        print(f"| {name} | {g('accuracy@1')} | {g('accuracy@3')} | {g('ndcg@10')} | {tt} |")

print("\n## 8단계 모드별 속도·메모리\n")
print("| 모드 | 검출 ms (최대 fps) | 임베딩 ms/개 | RAM (MB, RSS+GPU, 최고) | 추출 수 |")
print("| --- | --- | --- | --- | --- |")
for mode in ("yolo", "relabel", "yoloe"):
    p = f"runs/stage8/{mode}.json"
    if not os.path.exists(p):
        continue
    try:
        r = json.load(open(p))
    except json.JSONDecodeError:
        print(f"| {mode} | (결과 없음) | | | |"); continue
    s = r["stats"]
    print(f"| {mode} | {s['det_ms']:.0f} ({s['det_fps_max']}) | {s['embed_ms']:.0f} | "
          f"{s['rss_peak_mb']}+{s['gpu_peak_mb']} | {r['total']} |")
for mode in ("yolo", "relabel", "yoloe"):
    p = f"runs/stage8/{mode}.json"
    if os.path.exists(p):
        try:
            items = json.load(open(p))["items"]
        except json.JSONDecodeError:
            continue
        print(f"\n[{mode}] " + ", ".join(f"{it['img'].split('/')[-1]}→{it['label']}({it['lscore']}, det {it['conf']})"
                                       for it in sorted(items, key=lambda x: x["img"])))

print("\n## 9단계 영상별 동작 횟수 (raw vs 안정화)\n")
print("| 영상 | raw 횟수 | 안정화 횟수 | 행동 요약 (사람 단위) |")
print("| --- | --- | --- | --- |")
for p in sorted(glob.glob("runs/stage9/*_raw.json")):
    name = os.path.basename(p)[:-9]
    try:
        raw = json.load(open(p))
        sm = json.load(open(f"runs/stage9/{name}_smooth.json"))
    except (OSError, json.JSONDecodeError):
        print(f"| {name} | (결과 없음) | | |"); continue
    act = lambda c: ", ".join(f"{k} {v}" for k, v in c.items()
                              if k in ("만세", "한팔들기", "손흔들기", "악수")) or "-"
    print(f"| {name.split('-')[0]} | {act(raw['counts'])} | {act(sm['counts'])} | "
          f"{', '.join(f'{k} {v}' for k, v in sm['beh'].items()) or '-'} |")
