"""Szomorú, pörgős emo-rap beat generátor (csak numpy + ffmpeg).

Futtatás:  python3 make_beat.py   ->  sad_bounce_150bpm.wav / .mp3
Hangnem: A-moll, 150 BPM, progresszió: Am - F - C - G
"""
import numpy as np, wave, subprocess, os

SR = 44100
BPM = 150
BEAT = 60 / BPM
BAR = BEAT * 4
rng = np.random.default_rng(7)

def note_freq(n):  # MIDI -> Hz
    return 440.0 * 2 ** ((n - 69) / 12)

# ---------- hangszerek ----------
def pluck(freq, dur, bright=0.5):
    """Karplus-Strong gitár-pengetés."""
    n = int(dur * SR)
    p = max(2, int(SR / freq))
    buf = rng.uniform(-1, 1, p)
    # lágyítjuk a kezdeti zajt (melegebb hang)
    for _ in range(int((1 - bright) * 4)):
        buf = 0.5 * (buf + np.roll(buf, 1))
    out = np.empty(n)
    decay = 0.996
    for i in range(n):
        j = i % p
        out[i] = buf[j]
        buf[j] = decay * 0.5 * (buf[j] + buf[(j + 1) % p])
    return out

def sine_env(t, a, d):
    return np.minimum(t / a, 1.0) * np.exp(-t / d)

def make_808(freq, dur, glide_from=None):
    t = np.arange(int(dur * SR)) / SR
    f = np.full_like(t, freq)
    if glide_from:
        g = np.exp(-t / 0.06)
        f = freq + (glide_from - freq) * g
    phase = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(phase) * sine_env(t, 0.003, 0.9)
    x = np.tanh(2.2 * x) * 0.8  # kis torzítás, hogy telefonon is szóljon
    rel = np.clip((dur - t) / 0.03, 0, 1)
    return x * rel

def kick():
    t = np.arange(int(0.35 * SR)) / SR
    f = 50 + 120 * np.exp(-t / 0.03)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.12)
    click = rng.uniform(-1, 1, len(t)) * np.exp(-t / 0.002) * 0.3
    return np.tanh(1.8 * (x + click))

def clap():
    t = np.arange(int(0.4 * SR)) / SR
    noise = rng.uniform(-1, 1, len(t))
    # sávszűrés: differencia + simítás
    noise = np.diff(noise, prepend=0)
    noise = np.convolve(noise, np.ones(3) / 3, "same")
    env = np.zeros_like(t)
    for off in (0, 0.011, 0.022):
        tt = np.clip(t - off, 0, None)
        env += (t >= off) * np.exp(-tt / 0.008)
    env += (t >= 0.022) * np.exp(-np.clip(t - 0.022, 0, None) / 0.11) * 0.8
    body = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.04) * 0.4
    return (noise * env + body) * 0.7

def hat(open_=False):
    d = 0.25 if open_ else 0.05
    t = np.arange(int(d * SR)) / SR
    n = rng.uniform(-1, 1, len(t))
    n = n - np.convolve(n, np.ones(6) / 6, "same")  # felüláteresztő
    return n * np.exp(-t / (0.07 if open_ else 0.012)) * 0.22

def reverb(x, secs=2.2, wet=0.35):
    n = int(secs * SR)
    t = np.arange(n) / SR
    ir = rng.uniform(-1, 1, n) * np.exp(-t / (secs / 5))
    ir = np.convolve(ir, np.ones(20) / 20, "same")  # sötétebb lecsengés
    ir /= np.sqrt(np.sum(ir ** 2))
    L = len(x) + n
    N = 1 << (L - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(ir, N), N)[: len(x)]
    return (1 - wet) * x + wet * y * 0.6

def lowpass(x, alpha):
    y = np.empty_like(x); acc = 0.0
    for i, v in enumerate(x):
        acc += alpha * (v - acc); y[i] = acc
    return y

# ---------- zene ----------
# Am, F, C, G  (MIDI hangok, gitár-fekvés)
CHORDS = [
    [57, 64, 69, 72, 76],  # Am
    [53, 60, 65, 69, 72],  # F
    [48, 55, 60, 64, 67],  # C
    [55, 59, 62, 67, 71],  # G
]
BASS = [45, 41, 36, 43]  # A1, F1, C1, G1 ... 808 gyökhangok (oktávval lejjebb)
ARP = [0, 2, 3, 4, 3, 2, 1, 3]  # nyolcados arpeggio minta (index a chordban)

SECTIONS = (  # (név, ütemszám, gitár, dob, 808)
    [("intro", 4, True, False, False),
     ("hook", 8, True, True, True),
     ("verse", 16, True, True, True),
     ("hook", 8, True, True, True),
     ("verse", 16, True, True, True),
     ("hook", 8, True, True, True),
     ("outro", 4, True, False, False)]
)
total_bars = sum(s[1] for s in SECTIONS)
length = int((total_bars * BAR + 3) * SR)
gtr = np.zeros(length); drums = np.zeros(length); bass = np.zeros(length)

def put(buf, sig, t, gain=1.0):
    i = int(t * SR)
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[: j - i] * gain

pluck_cache = {}
def get_pluck(n):
    if n not in pluck_cache:
        pluck_cache[n] = pluck(note_freq(n), 1.6, bright=0.6)
    return pluck_cache[n]

K, C, H, HO = kick(), clap(), hat(), hat(True)

bar_idx = 0
for name, bars, g_on, d_on, b_on in SECTIONS:
    for b in range(bars):
        t0 = bar_idx * BAR
        ci = b % 4
        chord = CHORDS[ci]
        # --- gitár arpeggio ---
        if g_on:
            for s, idx in enumerate(ARP):
                n = chord[idx] + (12 if (name == "hook" and s in (3, 4)) else 0)
                vel = 0.55 if s % 2 else 0.7
                put(gtr, get_pluck(n), t0 + s * BEAT / 2 + rng.uniform(0, 0.006), vel)
        # --- dobok ---
        if d_on:
            # kick minta (bouncy, nem sima four-on-floor)
            kick_pos = [0, 1.75, 2.5] if b % 2 == 0 else [0, 0.75, 2.75, 3.5]
            for p in kick_pos:
                put(drums, K, t0 + p * BEAT, 0.9)
            put(drums, C, t0 + 2 * BEAT, 0.8)            # clap a 3-ra (half-time)
            if b % 4 == 3:
                put(drums, C, t0 + 3.75 * BEAT, 0.35)    # ghost clap
            # hi-hatek: nyolcadok + rollok
            step = 0
            pos = 0.0
            roll = (b % 4 == 3) or (name == "hook" and b % 2 == 1)
            while pos < 4:
                if roll and 2.5 <= pos < 3.0:
                    sub = 1 / 6       # triolás roll
                elif roll and 3.5 <= pos < 4:
                    sub = 1 / 8       # 32-es roll
                elif name == "verse" and pos >= 3 and b % 8 == 7:
                    sub = 1 / 3       # triolás flow-váltás
                else:
                    sub = 0.5
                g = 0.55 + 0.25 * (step % 2 == 0)
                put(drums, H, t0 + pos * BEAT, g)
                pos += sub; step += 1
            if b % 2 == 1:
                put(drums, HO, t0 + 3.5 * BEAT, 0.25)
        # --- 808 ---
        if b_on:
            root = note_freq(BASS[ci] - 12 + 12)  # A1~55Hz környéke
            pattern = [(0, 1.75), (1.75, 0.75), (2.5, 1.5)] if b % 2 == 0 else \
                      [(0, 0.75), (0.75, 2.0), (2.75, 0.75), (3.5, 0.5)]
            for k, (p, d) in enumerate(pattern):
                f = root
                glide = None
                if b % 4 == 3 and k == len(pattern) - 1:
                    f = root * 2 ** (2 / 12)          # felcsúszás a következő ütembe
                    glide = root
                put(bass, make_808(f, d * BEAT, glide), t0 + p * BEAT, 0.85)
        bar_idx += 1

# intro/outro: tompított, "telefonból szól" gitár
intro_end = int(4 * BAR * SR)
gtr[:intro_end] = lowpass(gtr[:intro_end], 0.12) * 1.6
out_start = int((total_bars - 4) * BAR * SR)
gtr[out_start:] = lowpass(gtr[out_start:], 0.12) * 1.6

gtr = reverb(gtr, 2.6, 0.4)
# sidechain-szerű "pumpálás" a kickre a gitáron
duck = np.ones(length)
bar_idx = 0
mix = 0.7 * gtr + 0.6 * drums + 0.8 * bass
mix = np.tanh(1.1 * mix)
mix /= np.max(np.abs(mix)) / 0.95

# fade out
fo = int(3 * SR)
mix[-fo:] *= np.linspace(1, 0, fo)

stereo = np.stack([mix, mix], axis=1)
# enyhe sztereó szélesítés a gitáron
d = int(0.012 * SR)
g_wide = 0.08 * np.roll(gtr, d) / (np.max(np.abs(gtr)) + 1e-9)
stereo[:, 0] += g_wide; stereo[:, 1] -= g_wide
stereo /= np.max(np.abs(stereo)) / 0.95

here = os.path.dirname(os.path.abspath(__file__))
wav_path = os.path.join(here, "sad_bounce_150bpm.wav")
with wave.open(wav_path, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((stereo * 32767).astype("<i2").tobytes())
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path,
                "-b:a", "256k", wav_path.replace(".wav", ".mp3")], check=True)
print("kész:", wav_path, f"{len(mix)/SR:.1f}s")
