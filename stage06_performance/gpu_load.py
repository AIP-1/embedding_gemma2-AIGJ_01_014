"""6단계 GPU 사용률 측정용: 모델 로드 후 인코딩만 10초간 반복 (그동안 tegrastats를 본다)
사용법: python gpu_load.py text | image | video | audio | all
"""
import glob, sys, time
import torch
torch.cuda.set_per_process_memory_fraction(0.45)   # 공용 보드: 약 3.3GB까지만 사용
from eg2 import load

mode = sys.argv[1] if len(sys.argv) > 1 else "text"
model = load(vision=mode in ("image", "video", "all"), audio=mode in ("audio", "all"))

texts = ["수업 중에 졸리고 머리가 맑지 않아요", "여름에 냉방 몇 도로 맞춰야 하나요?"]
images = [{"image": p} for p in sorted(glob.glob("media/*.jpg"))]
audios = [{"audio": p} for p in sorted(glob.glob("media/*.wav")) if "/raw_" not in p]

def run():
    if mode in ("text", "all"):  model.encode(texts, prompt_name="SearchQuery")
    if mode in ("image", "all"): model.encode(images)
    if mode == "video":          model.encode({"video": "media/clip.mp4"})
    if mode in ("audio", "all"): model.encode(audios)

run()                                            # 준비 실행 (값 버림)
print(f"[{mode}] 로드 완료. 지금부터 10초간 인코딩 반복 → tegrastats를 보세요", flush=True)
t_end, n = time.time() + 10, 0
while time.time() < t_end:
    run(); n += 1
print(f"[{mode}] 끝. 10초 동안 {n}회 반복 (1회 평균 {10000 / n:.0f} ms)")
print(f"[{mode}] GPU 최대 reserved: {torch.cuda.max_memory_reserved() / 2**20:.0f} MB")
