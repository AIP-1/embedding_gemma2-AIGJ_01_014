"""6단계 '전체(740M) / 혼합' 측정: 텍스트·이미지·음성을 한 모델로 인코딩, 2회 실행"""
import glob, time, torch
torch.cuda.set_per_process_memory_fraction(0.45)   # 공용 보드: 약 3.3GB까지만
from eg2 import load

t = time.perf_counter()
model = load(vision=True, audio=True)
print(f"로드: {time.perf_counter() - t:.1f} s")

texts = ["수업 중에 졸리고 머리가 맑지 않아요", "날카로운 경보음"]
images = [{"image": p} for p in sorted(glob.glob("media/*.jpg"))]
audios = [{"audio": p} for p in sorted(glob.glob("media/*.wav")) if "/raw_" not in p]

for run in (1, 2):
    torch.cuda.synchronize(); t = time.perf_counter()
    model.encode(texts, prompt_name="SearchQuery", normalize_embeddings=True)
    model.encode(images, normalize_embeddings=True)
    model.encode(audios, normalize_embeddings=True)
    torch.cuda.synchronize()
    print(f"[{run}회차] 텍스트 {len(texts)} + 이미지 {len(images)} + 음성 {len(audios)}: "
          f"{(time.perf_counter() - t) * 1000:.0f} ms")
