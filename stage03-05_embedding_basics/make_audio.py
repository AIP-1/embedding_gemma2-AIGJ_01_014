"""5단계 테스트 소리 생성 (ffmpeg lavfi 대신): 16 kHz 모노 WAV로 바로 저장"""
import numpy as np, soundfile as sf

SR = 16000
rng = np.random.default_rng(0)

# 경보음: 1 kHz 사인파 3초 (문서의 sine=frequency=1000:duration=3)
t = np.arange(3 * SR) / SR
beep = 0.5 * np.sin(2 * np.pi * 1000 * t)
sf.write("media/beep.wav", beep.astype(np.float32), SR, subtype="PCM_16")

# 빗소리 같은 잡음: 핑크 노이즈 5초 (문서의 anoisesrc=color=pink:duration=5)
n = 5 * SR
spec = np.fft.rfft(rng.standard_normal(n))
f = np.fft.rfftfreq(n, 1 / SR)
spec[1:] /= np.sqrt(f[1:])          # 1/f 세기 → 낮은 소리가 강한 '쏴-' 소리
spec[0] = 0
pink = np.fft.irfft(spec, n)
pink = 0.5 * pink / np.abs(pink).max()
sf.write("media/noise.wav", pink.astype(np.float32), SR, subtype="PCM_16")

for p in ["media/beep.wav", "media/noise.wav", "media/speech.wav", "media/mic.wav"]:
    try:
        info = sf.info(p)
        print(f"{p}: {info.samplerate} Hz, {info.channels} ch, {info.duration:.1f}초")
    except Exception as e:
        print(f"{p}: 읽기 실패 ({e})")
