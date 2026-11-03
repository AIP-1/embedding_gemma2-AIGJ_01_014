"""6단계: 내 프로세스만 측정 (공용 보드용)
사용법: python measure.py test_text.py
- CPU 최대 RSS : 이 프로세스가 쓴 일반 메모리의 최고값 (/proc/self/status의 VmHWM)
- GPU reserved : PyTorch가 GPU용으로 잡아 둔 메모리 최고값 (모델 가중치 + 연산 버퍼)
Jetson은 CPU·GPU가 같은 RAM을 쓰므로 두 값의 합이 이 프로그램이 차지한 RAM의 근사값이다.
"""
import runpy, sys, time, torch

def status(key):
    for line in open("/proc/self/status"):
        if line.startswith(key):
            return int(line.split()[1]) / 1024          # kB → MB
    return 0.0

script = sys.argv[1]
sys.argv = sys.argv[1:]                                 # 대상 스크립트가 자기 인자를 그대로 받게
t = time.perf_counter()
runpy.run_path(script, run_name="__main__")
total = time.perf_counter() - t

rss = status("VmHWM")
gpu = torch.cuda.max_memory_reserved() / 2**20 if torch.cuda.is_initialized() else 0.0
print("\n" + "=" * 48)
print(f"[measure] {script}")
print(f"  전체 실행 시간     : {total:.1f} s (모델 로드 포함)")
print(f"  CPU 최대 RSS       : {rss:,.0f} MB")
print(f"  GPU 최대 reserved  : {gpu:,.0f} MB")
print(f"  합계 (근사)        : {rss + gpu:,.0f} MB")
print("=" * 48)
