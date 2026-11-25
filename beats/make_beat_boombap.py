"""Sötét, poros boom bap beat (saját loop), csak numpy + ffmpeg.

Futtatás:  python3 make_beat_boombap.py  ->  boombap_dark_88bpm.mp3
Hangnem: D-moll (harmonikus), 88 BPM, swingelt 16-odok
Loop: Dm - Bb - Gm - A  (zongora riff + vonósok, "samplelt" hangzásra koszolva:
      lemezjátszó-imbolygás, 12 bites bitcrush, szűrés, bakelitsercegés)
Szerkezet: intro 4 | verze 16 | refrén 8 | verze 16 | refrén 8 | outro 4 (szalaglassulással)
"""
import numpy as np, wave, subprocess, os

SR = 44100
BPM = 88
BEAT = 60 / BPM
BAR = BEAT * 4
SWING = 0.6          # 0.5 = egyenes, 0.6 = boom bap swing
rng = np.random.default_rng(88)

def hz(n):
    return 440.0 * 2 ** ((n - 69) / 12)

def at(t0, p):
    """16-od pozíció (0..15) -> idő, swinggel."""
    return t0 + (p // 2) * (BEAT / 2) + (p % 2) * (BEAT / 2) * SWING

def lowpass(x, alpha):
    N = 1 << (len(x) + 1).bit_length()
    w = np.fft.rfftfreq(N)
    H = alpha / (1 - (1 - alpha) * np.exp(-2j * np.pi * w))
    return np.fft.irfft(np.fft.rfft(x, N) * H, N)[: len(x)]

def highpass(x, alpha):
    return x - lowpass(x, alpha)

def reverb(x, secs=1.5, wet=0.3, dark=20):
    n = int(secs * SR); t = np.arange(n) / SR
    ir = np.convolve(rng.uniform(-1, 1, n) * np.exp(-t / (secs / 5)), np.ones(dark) / dark, "same")
    ir /= np.sqrt(np.sum(ir ** 2))
    N = 1 << (len(x) + n).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(ir, N), N)[: len(x)]
    return (1 - wet) * x + wet * y * 0.6

def norm(x):
    return x / (np.max(np.abs(x)) + 1e-9)

# ---------- hangszerek ----------
def piano(n, dur, vel=1.0):
    t = np.arange(int(dur * SR)) / SR
    f = hz(n)
    x = np.zeros_like(t)
    for k in range(1, 10):
        fk = f * k * np.sqrt(1 + 0.0004 * k * k)
        if fk > 11000:
            break
        d = 2.6 / (1 + 0.55 * k) * (262 / f) ** 0.35
        x += (1 / k ** 1.1) * np.sin(2 * np.pi * fk * t + rng.uniform(0, 6.28)) * np.exp(-t / d)
    x += rng.normal(0, 1, len(t)) * np.exp(-t / 0.004) * 0.05
    env = np.clip(t / 0.002, 0, 1) * np.clip((dur - t) / 0.15, 0, 1)
    return x * env * vel

def strings(notes, dur):
    t = np.arange(int(dur * SR)) / SR
    x = np.zeros_like(t)
    for n in notes:
        for d in (0.993, 0.998, 1.003, 1.008):
            vib = 1 + 0.004 * np.sin(2 * np.pi * (5 + 2 * d) * t + 9 * d)
            ph = np.cumsum(hz(n) * d * vib) / SR
            x += 2 * (ph % 1.0) - 1
    x = lowpass(lowpass(x, 0.1), 0.1)
    env = np.clip(t / 0.35, 0, 1) * np.clip((dur - t) / 0.3, 0, 1)
    return x * env / len(notes)

def bass(n, dur):
    t = np.arange(int(dur * SR)) / SR
    f = hz(n)
    x = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t) + 0.1 * np.sin(6 * np.pi * f * t)
    x = np.tanh(1.5 * x) * np.exp(-t / 0.6)
    return x * np.clip(t / 0.005, 0, 1) * np.clip((dur - t) / 0.03, 0, 1)

def kick():
    t = np.arange(int(0.35 * SR)) / SR
    f = 52 + 90 * np.exp(-t / 0.03)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.16)
    knock = np.sin(2 * np.pi * 110 * t) * np.exp(-t / 0.03) * 0.4
    click = highpass(rng.uniform(-1, 1, len(t)), 0.3) * np.exp(-t / 0.003) * 0.5
    return np.tanh(2.2 * (x + knock + click))

def snare():
    t = np.arange(int(0.4 * SR)) / SR
    n = highpass(rng.uniform(-1, 1, len(t)), 0.08)
    n = lowpass(n, 0.45)
    crack = n * (np.exp(-t / 0.05) + 0.25 * np.exp(-t / 0.18))
    body = (np.sin(2 * np.pi * 190 * t) + 0.5 * np.sin(2 * np.pi * 330 * t)) * np.exp(-t / 0.06)
    return np.tanh(1.8 * (crack + 0.6 * body)) * 0.9

def rim():
    t = np.arange(int(0.08 * SR)) / SR
    return (np.sin(2 * np.pi * 520 * t) + highpass(rng.uniform(-1, 1, len(t)), 0.3) * 0.5) * np.exp(-t / 0.012) * 0.5

def hat(open_=False):
    d = 0.3 if open_ else 0.06
    t = np.arange(int(d * SR)) / SR
    n = highpass(rng.uniform(-1, 1, len(t)), 0.35)
    return n * np.exp(-t / (0.09 if open_ else 0.014)) * 0.35

def crackle(n_samples):
    x = np.zeros(n_samples)
    k = rng.poisson(18 * n_samples / SR)
    idx = rng.integers(0, n_samples, k)
    x[idx] = rng.normal(0, 1, k) * rng.uniform(0.2, 1.0, k) ** 3
    x = lowpass(x, 0.5)
    hiss = lowpass(rng.normal(0, 1, n_samples), 0.25) * 0.015
    rumble = lowpass(rng.normal(0, 1, n_samples), 0.002) * 0.3
    return x + hiss + rumble

def lofi(x, cutoff):
    """'Samplelt' koszolás: lemezjátszó-imbolygás, sávszűrés, 12 bit, sample-and-hold."""
    t = np.arange(len(x)) / SR
    warp = t + 0.0018 * np.sin(2 * np.pi * 0.45 * t) + 0.00025 * np.sin(2 * np.pi * 5.5 * t)
    x = np.interp(warp * SR, np.arange(len(x)), x)
    x = lowpass(highpass(x, 0.012), cutoff)
    x = np.round(norm(x) * 2047) / 2047
    x = np.repeat(x[::2], 2)[: len(x)]
    return lowpass(x, 0.6)

# ---------- zene ----------
# Zongora riff: (16-od kezdés, hossz 16-odban, MIDI) ütemenként
RIFF = [
    [(0, 2, 69), (2, 2, 74), (4, 4, 77), (8, 2, 76), (10, 2, 74), (12, 4, 69)],   # Dm: A D F. E D A.
    [(0, 2, 70), (2, 2, 74), (4, 4, 77), (8, 2, 76), (10, 2, 74), (12, 4, 70)],   # Bb: B D F. E D B.
    [(0, 2, 67), (2, 2, 70), (4, 4, 74), (8, 2, 72), (10, 2, 70), (12, 4, 67)],   # Gm: G B D. C B G.
    [(0, 2, 69), (2, 2, 73), (4, 4, 76), (8, 2, 74), (10, 2, 73), (12, 2, 69), (14, 2, 64)],  # A: A Cisz E. D Cisz A E
]
LH = [38, 34, 31, 33]                                    # bal kéz / basszus gyökök: D2 Bb1 G1 A1
PADS = [[62, 65, 69], [62, 65, 70], [62, 67, 70], [61, 64, 69]]
HIGH = [81, 77, 79, 76]                                  # refrén: magas vonós ellenszólam A5 F5 G5 E5

SECTIONS = [("intro", 4), ("verse", 16), ("hook", 8), ("verse", 16), ("hook", 8), ("outro", 4)]
total_bars = sum(b for _, b in SECTIONS)
L = int((total_bars * BAR + 3) * SR)
smp_v, smp_h, drums, bas = (np.zeros(L) for _ in range(4))
sec_of_sample = np.zeros(L, dtype=int)  # 0 intro/outro, 1 verse, 2 hook  (szűrő nyitáshoz)

def put(buf, sig, t, g=1.0):
    i = int(t * SR); j = min(len(buf), i + len(sig))
    if j > i:
        buf[i:j] += sig[: j - i] * g

K, S, R, H, HO = kick(), snare(), rim(), hat(), hat(True)
cache = {}
def pno(n, d, v):
    k = (n, round(d, 3), v)
    if k not in cache:
        cache[k] = piano(n, d, v)
    return cache[k]

S16 = BEAT / 4
bi = 0
for si, (name, bars) in enumerate(SECTIONS):
    nxt = SECTIONS[si + 1][0] if si + 1 < len(SECTIONS) else None
    for b in range(bars):
        t0 = bi * BAR
        ci = b % 4
        tgt = smp_h if name == "hook" else smp_v
        sec_of_sample[int(t0 * SR):int((t0 + BAR) * SR)] = {"intro": 0, "outro": 0, "verse": 1, "hook": 2}[name]

        # --- a "sample": zongora riff + bal kéz + vonósok ---
        for st, ln, n in RIFF[ci]:
            put(tgt, pno(n, max(ln * S16 * 1.6, 0.45), 0.8 if st % 4 == 0 else 0.65), at(t0, st))
        put(tgt, pno(LH[ci] + 12, BAR, 0.7), t0)
        put(tgt, strings(PADS[ci], BAR + 0.3), t0, 0.5)
        if name == "hook":
            put(tgt, strings([HIGH[ci]], BAR + 0.3), t0, 0.45)

        # --- dobok ---
        if name in ("verse", "hook"):
            section_end = b == bars - 1
            phrase_end = b % 8 == 7
            kp = [0, 7, 10] if b % 2 == 0 else [0, 6, 10, 13]
            for p in kp:
                if not (phrase_end and p >= 12):
                    put(drums, K, at(t0, p), 0.95)
            put(drums, S, at(t0, 4), 0.9)
            if not phrase_end:
                put(drums, S, at(t0, 12), 0.9)
            else:
                put(drums, S, at(t0, 12), 0.9)
                put(drums, S, at(t0, 15), 0.45)        # kis felvezetés a következő 8 előtt
            if b % 4 == 3:
                put(drums, R, at(t0, 15), 0.6)         # ghost rim
            for p in range(16):
                if section_end and p >= 12:
                    break
                if p % 2 == 0:
                    put(drums, H, at(t0, p), 0.75 if p % 4 == 0 else 0.55)
                elif p in (3, 11) or (name == "hook" and p in (7, 15)):
                    put(drums, H, at(t0, p), 0.3)      # szellem 16-odok
            if b % 2 == 1:
                put(drums, HO, at(t0, 14), 0.35)

            # --- basszus a kick ritmusára ---
            for p, ln in ((0, 6), (7 if b % 2 == 0 else 6, 3), (10, 5)):
                if not (phrase_end and p >= 12):
                    put(bas, bass(LH[ci], ln * S16), at(t0, p), 0.9)
        bi += 1

# --- feldolgozás ---
# a sample szűrője szekciónként nyílik (intro tompa -> verze sötét -> refrén nyitottabb)
sample = smp_v + smp_h
dark = lofi(sample, 0.05)
mid = lofi(sample, 0.11)
bright = lofi(sample, 0.25)
w = np.stack([sec_of_sample == 0, sec_of_sample == 1, sec_of_sample == 2]).astype(float)
k = int(0.05 * SR)
w = np.array([np.convolve(r, np.ones(k) / k, "same") for r in w])  # puha átmenet
sample = w[0] * dark * 1.4 + w[1] * mid + w[2] * bright
sample = reverb(sample, 1.8, 0.25)

drums_p = reverb(drums, 0.7, 0.18, dark=8)
drums_p = np.tanh(1.6 * norm(drums_p))
drums_p = np.round(drums_p * 4095) / 4095        # kicsit koszos, 13 bites dob

mix = 0.55 * norm(sample) + 0.62 * drums_p + 0.5 * norm(lowpass(bas, 0.06)) + 0.05 * crackle(L)
mix = np.tanh(1.1 * mix)

# --- outro vége: szalaglassulás (tape stop) az utolsó ütemen ---
end = int(total_bars * BAR * SR)
ts = int(BAR * SR)
seg_start = end - ts
rate = np.linspace(1, 0, ts) ** 0.7
pos = seg_start + np.cumsum(rate)
mix[seg_start:end] = np.interp(pos, np.arange(L), mix) * np.linspace(1, 0.2, ts)
mix = mix[: end + int(0.3 * SR)]
mix[end:] *= np.linspace(0.2, 0, len(mix) - end)

d = int(0.013 * SR)
wide = 0.08 * np.roll(norm(sample), d)[: len(mix)]
st = np.stack([mix + wide, mix - wide], axis=1)
st /= np.max(np.abs(st)) / 0.95

here = os.path.dirname(os.path.abspath(__file__))
wav = os.path.join(here, "boombap_dark_88bpm.wav")
with wave.open(wav, "wb") as f:
    f.setnchannels(2); f.setsampwidth(2); f.setframerate(SR)
    f.writeframes((st * 32767).astype("<i2").tobytes())
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-b:a", "256k",
                wav.replace(".wav", ".mp3")], check=True)
os.remove(wav)
print("kész:", wav.replace(".wav", ".mp3"), f"{len(mix)/SR:.1f}s")
