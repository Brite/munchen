"""Lassabb, érzelgős emo-trap beat FÜLBEMÁSZÓ, dúdolható dallammal (saját szerzemény).

Futtatás:  python3 make_beat_hum.py  ->  hum_hook_136bpm.mp3
Hangnem: C-moll, 136 BPM (half-time érzés ~68), progresszió: Cm - Ab - Eb - Bb
Lead: emberi "ooh/oh" dúdolás-szerű hang (formáns-szintézis), zongora, pad, 808, trap dobok.

Dallam-logika (hogy megmaradjon):
  - egy rövid motívum (G - Esz^ - D - C): kis ugrás fel, aztán lépegetve le
  - a motívum 4 ütemenként visszatér, a 2. körben magasabb csúcsra fut (G5)
  - már az introban elhangzik, a verzékben halk visszhangként szól, a bridge után nagyban jön vissza
"""
import numpy as np, wave, subprocess, os

SR = 44100
BPM = 136
BEAT = 60 / BPM
BAR = BEAT * 4
E = BEAT / 2
rng = np.random.default_rng(23)

def hz(n):
    return 440.0 * 2 ** ((n - 69) / 12)

def lowpass(x, alpha):
    N = 1 << (len(x) + 1).bit_length()
    w = np.fft.rfftfreq(N)
    H = alpha / (1 - (1 - alpha) * np.exp(-2j * np.pi * w))
    return np.fft.irfft(np.fft.rfft(x, N) * H, N)[: len(x)]

def reverb(x, secs=2.5, wet=0.35, dark=20):
    n = int(secs * SR); t = np.arange(n) / SR
    ir = np.convolve(rng.uniform(-1, 1, n) * np.exp(-t / (secs / 5)), np.ones(dark) / dark, "same")
    ir /= np.sqrt(np.sum(ir ** 2))
    N = 1 << (len(x) + n).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(ir, N), N)[: len(x)]
    return (1 - wet) * x + wet * y * 0.6

def delay(x, secs, fb=0.3, mix=0.25):
    d = int(secs * SR); y = x.copy()
    for k in range(1, 5):
        y[d * k:] += x[: len(x) - d * k] * (fb ** k)
    return (1 - mix) * x + mix * y

def norm(x):
    return x / (np.max(np.abs(x)) + 1e-9)

# ---------- hangszerek ----------
VOWELS = {  # formánsok: (frekvencia, sávszélesség, erősség)
    "oo": [(330, 120, 1.0), (820, 140, 0.45), (2300, 200, 0.10)],
    "oh": [(470, 140, 1.0), (850, 150, 0.6), (2600, 220, 0.15)],
    "ah": [(720, 160, 1.0), (1150, 170, 0.7), (2700, 240, 0.2)],
}

def hum(n, dur, vowel="oo", detune=0.0, vib_phase=0.0):
    """Emberi dúdolás-szerű hang: additív szintézis formánsokkal, becsúszással, késleltetett vibratóval."""
    t = np.arange(int(dur * SR)) / SR
    f0 = hz(n) * 2 ** (detune / 1200)
    vib = 1 + 0.011 * np.sin(2 * np.pi * 5.0 * t + vib_phase) * np.clip((t - 0.12) / 0.3, 0, 1)
    scoop = 2 ** (-0.9 * np.exp(-t / 0.045) / 12)
    phase = 2 * np.pi * np.cumsum(f0 * vib * scoop) / SR
    x = np.zeros_like(t)
    for k in range(1, int(5000 // f0) + 1):
        fk = k * f0
        g = sum(a / (1 + ((fk - F) / bw) ** 2) for F, bw, a in VOWELS[vowel]) + 0.015
        x += (g / k) * np.sin(k * phase)
    breath = rng.normal(0, 1, len(t))
    breath = lowpass(breath, 0.2) - lowpass(breath, 0.03)
    x = norm(x) + 0.025 * breath
    env = np.clip(t / 0.05, 0, 1) * np.clip((dur - t) / 0.09, 0, 1)
    return x * env

def choir_hum(n, dur, vowel="oo"):
    """Két kissé elhangolt hang együtt: teltebb, 'többen dúdolják' érzés."""
    return 0.55 * hum(n, dur, vowel, -6, 0.0) + 0.55 * hum(n, dur, vowel, +6, 1.7)

def piano(n, dur, vel=1.0):
    t = np.arange(int(dur * SR)) / SR
    f = hz(n)
    x = np.zeros_like(t)
    for k in range(1, 10):
        fk = f * k * np.sqrt(1 + 0.0004 * k * k)
        if fk > 11000:
            break
        d = 2.8 / (1 + 0.55 * k) * (262 / f) ** 0.35
        x += (1 / k ** 1.1) * np.sin(2 * np.pi * fk * t + rng.uniform(0, 6.28)) * np.exp(-t / d)
    x += rng.normal(0, 1, len(t)) * np.exp(-t / 0.004) * 0.04
    env = np.clip(t / 0.002, 0, 1) * np.clip((dur - t) / 0.2, 0, 1)
    return lowpass(x, 0.35 + 0.3 * vel) * env * vel

def bell(n, dur=1.5):
    t = np.arange(int(dur * SR)) / SR
    f = hz(n)
    x = np.sin(2 * np.pi * f * t + 1.2 * np.sin(2 * np.pi * f * 3.5 * t) * np.exp(-t / 0.15))
    return x * np.exp(-t / 0.45)

def saw(f, t):
    return 2 * ((np.cumsum(np.broadcast_to(f, t.shape)) / SR) % 1.0) - 1

def pad(notes, dur):
    t = np.arange(int(dur * SR)) / SR
    x = sum(saw(hz(n) * d, t) for n in notes for d in (0.995, 1.005))
    env = np.clip(t / 0.5, 0, 1) * np.clip((dur - t) / 0.5, 0, 1)
    return lowpass(x, 0.03) * env / len(notes)

def make_808(freq, dur, glide_from=None):
    t = np.arange(int(dur * SR)) / SR
    f = np.full_like(t, freq)
    if glide_from:
        f = freq + (glide_from - freq) * np.exp(-t / 0.09)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(t / 0.004, 1) * np.exp(-t / 2.2)
    return np.tanh(2.4 * x) * np.clip((dur - t) / 0.04, 0, 1)

def kick():
    t = np.arange(int(0.3 * SR)) / SR
    f = 46 + 130 * np.exp(-t / 0.028)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.1)
    return np.tanh(2 * (x + rng.uniform(-1, 1, len(t)) * np.exp(-t / 0.002) * 0.25))

def snare():
    t = np.arange(int(0.45 * SR)) / SR
    n = np.diff(rng.uniform(-1, 1, len(t)), prepend=0)
    env = sum((t >= o) * np.exp(-np.clip(t - o, 0, None) / 0.012) for o in (0, 0.009, 0.018))
    tail = np.exp(-t / 0.13) * 0.6
    body = np.sin(2 * np.pi * 185 * t) * np.exp(-t / 0.05) * 0.5
    return (n * (env + tail) + body) * 0.7

def hat(open_=False):
    d = 0.25 if open_ else 0.05
    t = np.arange(int(d * SR)) / SR
    n = rng.uniform(-1, 1, len(t))
    n = n - np.convolve(n, np.ones(6) / 6, "same")
    return n * np.exp(-t / (0.07 if open_ else 0.011)) * 0.2

def riser(dur):
    t = np.arange(int(dur * SR)) / SR
    n = rng.normal(0, 1, len(t))
    n = n - lowpass(n, 0.05)
    return n * (t / dur) ** 2.5

# ---------- zene ----------
SCALE = [0, 2, 3, 5, 7, 8, 10]  # C-moll

def third_below(n):
    pc, octv = n % 12, n // 12
    i = SCALE.index(pc)
    j = i - 2
    return (octv + (j // 7)) * 12 + SCALE[j % 7]

CHORD_LH = [36, 32, 39, 34]                 # C2 Ab1 Eb2 Bb1 (zongora bal kéz, 1 oktávval feljebb szól)
CHORD_RH = [[55, 60, 63], [56, 60, 63], [55, 58, 63], [53, 58, 62]]
BASS = [36, 32, 39, 34]                     # 808 gyökhangok

# HOOK: (kezdés nyolcadban, hossz nyolcadban, MIDI hang) — 8 ütem
HOOK = [
    [(0, 1, 67), (1, 2, 75), (3, 1, 74), (4, 3, 72), (7, 1, 70)],   # Cm:  G  Esz^ D  C.. B
    [(0, 2, 72), (2, 1, 70), (3, 1, 68), (4, 4, 67)],               # Ab:  C  B  Asz G...
    [(0, 1, 67), (1, 2, 75), (3, 1, 74), (4, 2, 75), (6, 2, 77)],   # Eb:  G  Esz^ D  Esz F
    [(0, 3, 74), (3, 1, 72), (4, 4, 70)],                           # Bb:  D.. C  B...
    [(0, 1, 67), (1, 2, 75), (3, 1, 74), (4, 3, 72), (7, 1, 70)],   # Cm
    [(0, 2, 72), (2, 1, 70), (3, 1, 68), (4, 4, 67)],               # Ab
    [(0, 1, 67), (1, 2, 75), (3, 1, 74), (4, 2, 75), (6, 2, 79)],   # Eb:  ... csúcs: G5
    [(0, 2, 77), (2, 2, 75), (4, 4, 74)],                           # Bb:  F  Esz  D...
]
ECHO = [(4, 1, 67), (5, 1, 75), (6, 2, 74)]  # a motívum eleje, halk visszhang a verzékben
BRIDGE = [[(0, 8, 75)], [(0, 8, 72)], [(0, 8, 70)], [(0, 6, 74)]]

SECTIONS = [("intro", 4), ("hook", 8), ("verse", 16), ("hook", 8),
            ("bridge", 4), ("hook2", 8), ("outro", 5)]
total_bars = sum(b for _, b in SECTIONS)
L = int((total_bars * BAR + 4) * SR)
B = {k: np.zeros(L) for k in ("lead", "harm", "piano", "pad", "bell", "drums", "bass", "fx", "intro")}
kick_times = []

def put(buf, sig, t, g=1.0):
    i = int(t * SR); j = min(len(buf), i + len(sig))
    if j > i:
        buf[i:j] += sig[: j - i] * g

K, S, H, HO = kick(), snare(), hat(), hat(True)
cache = {}
def cached(fn, *a):
    key = (fn.__name__,) + a
    if key not in cache:
        cache[key] = fn(*a)
    return cache[key]

bi = 0
for si, (name, bars) in enumerate(SECTIONS):
    nxt = SECTIONS[si + 1][0] if si + 1 < len(SECTIONS) else None
    for b in range(bars):
        t0 = bi * BAR
        ci = b % 4
        last_bar = b == bars - 1
        into_hook = last_bar and nxt in ("hook", "hook2")
        full = name in ("hook", "verse", "hook2")

        # --- zongora (3+3+2 ritmus: 1, 2.5, 4. ütés) ---
        pv = 0.55 if name in ("intro", "bridge", "outro") else 0.75
        final = name == "outro" and b == bars - 1
        put(B["piano"], cached(piano, CHORD_LH[ci] + 12, BAR * (2 if final else 1), pv), t0)
        for p, v in ((0, 1.0), (1.5, 0.6), (3, 0.5)):
            if final and p > 0:
                break
            for n in CHORD_RH[ci]:
                put(B["piano"], cached(piano, n, BAR * 2 if final else BEAT * 1.6, pv * v), t0 + p * BEAT + rng.uniform(0, 0.01))

        if name in ("hook", "hook2", "bridge"):
            put(B["pad"], pad(CHORD_RH[ci] + [CHORD_LH[ci] + 12], BAR + 0.5), t0)

        # --- dúdolós dallam ---
        if name in ("intro", "hook", "hook2") or (name == "outro" and b < 4):
            phrase = HOOK[(b % 8) if name != "outro" else b]
            vowel = "oh" if name == "hook2" else "oo"
            target = "intro" if name in ("intro", "outro") else "lead"
            for st, ln, n in phrase:
                d = ln * E * 1.02
                put(B[target], cached(choir_hum, n, d, vowel), t0 + st * E)
                if name == "hook2":
                    put(B["harm"], cached(hum, third_below(n), d, "oo"), t0 + st * E)
                    put(B["bell"], cached(bell, n + 12), t0 + st * E, 0.35)
        elif name == "verse" and b % 2 == 1:
            for st, ln, n in ECHO:
                put(B["bell"], cached(bell, n), t0 + st * E, 0.5)
        elif name == "bridge":
            for st, ln, n in BRIDGE[b]:
                put(B["lead"], cached(choir_hum, n, ln * E, "ah"), t0 + st * E, 0.8)

        if into_hook:
            put(B["fx"], riser(BAR), t0)

        # --- dobok + 808 (half-time) ---
        if full:
            cut = 3.0 if into_hook else 99   # az utolsó ütés néma a hook előtt -> "drop"
            kp = [0, 2.5] if b % 2 == 0 else [0, 0.75, 2.5, 3.25]
            for p in kp:
                if p < cut:
                    put(B["drums"], K, t0 + p * BEAT, 0.9); kick_times.append(t0 + p * BEAT)
            if 2 < cut:
                put(B["drums"], S, t0 + 2 * BEAT, 0.8)
            pos, step = 0.0, 0
            while pos < min(4, cut):
                if name != "verse" and b % 4 == 3 and 3.0 <= pos < 3.5:   sub = 1 / 6
                elif name == "verse" and b % 4 == 1 and pos >= 2:        sub = 1 / 3
                elif b % 8 == 7 and pos >= 3.5:                          sub = 1 / 8
                else:                                                    sub = 0.5
                put(B["drums"], H, t0 + pos * BEAT, 0.45 + 0.3 * (step % 2 == 0))
                pos += sub; step += 1
            if b % 4 == 1:
                put(B["drums"], HO, t0 + 1.5 * BEAT, 0.3)

            r = hz(BASS[ci])
            pat = [(0, 2.5), (2.75, 1.0)] if b % 2 == 0 else [(0, 1.5), (1.5, 1.0), (2.5, 1.25)]
            for k, (p, d) in enumerate(pat):
                if p >= cut:
                    continue
                f, gl = r, None
                if b % 4 == 3 and k == len(pat) - 1:
                    f, gl = r * 2, r              # oktáv-csúszás a kör végén
                put(B["bass"], make_808(f, min(d, cut - p) * BEAT, gl), t0 + p * BEAT)
        bi += 1

# --- effektek ---
duck = np.ones(L)
for tk in kick_times:
    i = int(tk * SR); n = int(0.25 * SR)
    seg = 1 - 0.35 * np.exp(-np.arange(n) / SR / 0.09)
    duck[i:i + n] = np.minimum(duck[i:i + n], seg[: len(duck[i:i + n])])

lead = reverb(delay(B["lead"], BEAT * 0.75, 0.3, 0.22), 3.0, 0.38)
intro_l = reverb(lowpass(B["intro"], 0.06) * 2.0, 3.0, 0.5)   # "másik szobából szól" hangzás
harm = reverb(B["harm"], 3.0, 0.5)
piano_ = reverb(B["piano"], 2.4, 0.35) * duck
pad_ = reverb(B["pad"], 3.0, 0.5) * duck
bell_ = reverb(delay(B["bell"], BEAT * 0.75, 0.35, 0.3), 2.5, 0.5)
fx = reverb(B["fx"], 2.0, 0.4)

mus = (0.55 * norm(lead) + 0.35 * norm(intro_l) + 0.18 * norm(harm) + 0.45 * norm(piano_)
       + 0.18 * norm(pad_) + 0.12 * norm(bell_) + 0.12 * norm(fx))
mix = np.tanh(1.15 * (mus + 0.42 * norm(B["drums"]) + 0.7 * norm(B["bass"])))
end = int((total_bars * BAR + 3.5) * SR)
mix = mix[:end]
fo = int(4 * SR); mix[-fo:] *= np.linspace(1, 0, fo) ** 1.5

d = int(0.015 * SR)
w = (0.1 * np.roll(norm(piano_) + norm(pad_) + 0.5 * norm(harm), d))[:end]
st = np.stack([mix + w, mix - w], axis=1)
st /= np.max(np.abs(st)) / 0.95

here = os.path.dirname(os.path.abspath(__file__))
wav = os.path.join(here, "hum_hook_136bpm.wav")
with wave.open(wav, "wb") as f:
    f.setnchannels(2); f.setsampwidth(2); f.setframerate(SR)
    f.writeframes((st * 32767).astype("<i2").tobytes())
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-b:a", "256k",
                wav.replace(".wav", ".mp3")], check=True)
os.remove(wav)
print("kész:", wav.replace(".wav", ".mp3"), f"{len(mix)/SR:.1f}s")
