"""Sötét, pörgős emo-trap beat (saját dallam), csak numpy + ffmpeg.

Futtatás:  python3 make_beat_dark.py  ->  dark_bounce_155bpm.mp3
Hangnem: F-moll, 155 BPM, progresszió: Fm - Db - Bbm - C
Hangszerek: lebegő, detunolt synth lead, sötét pad, harang-arpeggio, 808, trap dobok.
"""
import numpy as np, wave, subprocess, os

SR = 44100
BPM = 155
BEAT = 60 / BPM
BAR = BEAT * 4
rng = np.random.default_rng(11)

def hz(n):
    return 440.0 * 2 ** ((n - 69) / 12)

def lowpass(x, alpha):
    # egypólusú szűrő vektorizálva FFT-vel (gyors)
    N = 1 << (len(x) + 1).bit_length()
    w = np.fft.rfftfreq(N)
    H = alpha / (1 - (1 - alpha) * np.exp(-2j * np.pi * w))
    return np.fft.irfft(np.fft.rfft(x, N) * H, N)[: len(x)]

def saw(f, t):
    ph = np.cumsum(np.broadcast_to(f, t.shape)) / SR
    return 2 * (ph % 1.0) - 1

def lead(n, dur):
    """Detunolt szuperszaw + vibrato, kicsit 'sírós' hang."""
    t = np.arange(int(dur * SR)) / SR
    vib = 1 + 0.006 * np.sin(2 * np.pi * 5.5 * t) * np.clip(t / 0.25, 0, 1)
    f = hz(n) * vib
    x = sum(saw(f * d, t) for d in (0.993, 1.0, 1.007)) / 3
    x += 0.4 * np.sin(2 * np.pi * np.cumsum(f) / SR * 0.5)  # al-oktáv a testért
    env = np.clip(t / 0.01, 0, 1) * np.clip((dur - t) / 0.04, 0, 1) * (0.75 + 0.25 * np.exp(-t / 0.15))
    return lowpass(x, 0.18) * env

def bell(n, dur=1.2):
    t = np.arange(int(dur * SR)) / SR
    f = hz(n)
    x = np.sin(2 * np.pi * f * t + 1.8 * np.sin(2 * np.pi * f * 3.5 * t) * np.exp(-t / 0.2))
    return x * np.exp(-t / 0.35)

def pad(notes, dur):
    t = np.arange(int(dur * SR)) / SR
    x = sum(saw(hz(n) * d, t) for n in notes for d in (0.996, 1.004))
    env = np.clip(t / 0.4, 0, 1) * np.clip((dur - t) / 0.4, 0, 1)
    return lowpass(x, 0.04) * env / len(notes)

def make_808(freq, dur, glide_from=None):
    t = np.arange(int(dur * SR)) / SR
    f = np.full_like(t, freq)
    if glide_from:
        f = freq + (glide_from - freq) * np.exp(-t / 0.07)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(t / 0.003, 1) * np.exp(-t / 1.1)
    return np.tanh(3.0 * x) * 0.8 * np.clip((dur - t) / 0.03, 0, 1)

def kick():
    t = np.arange(int(0.3 * SR)) / SR
    f = 48 + 140 * np.exp(-t / 0.025)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.1)
    return np.tanh(2 * (x + rng.uniform(-1, 1, len(t)) * np.exp(-t / 0.002) * 0.3))

def snare():
    t = np.arange(int(0.35 * SR)) / SR
    n = np.diff(rng.uniform(-1, 1, len(t)), prepend=0)
    body = np.sin(2 * np.pi * 200 * t) * np.exp(-t / 0.05)
    return (n * np.exp(-t / 0.09) * 0.7 + body * 0.5) * 0.8

def hat(open_=False):
    d = 0.25 if open_ else 0.05
    t = np.arange(int(d * SR)) / SR
    n = rng.uniform(-1, 1, len(t))
    n = n - np.convolve(n, np.ones(6) / 6, "same")
    return n * np.exp(-t / (0.07 if open_ else 0.012)) * 0.22

def reverb(x, secs=2.5, wet=0.35):
    n = int(secs * SR); t = np.arange(n) / SR
    ir = np.convolve(rng.uniform(-1, 1, n) * np.exp(-t / (secs / 5)), np.ones(20) / 20, "same")
    ir /= np.sqrt(np.sum(ir ** 2))
    N = 1 << (len(x) + n).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(ir, N), N)[: len(x)]
    return (1 - wet) * x + wet * y * 0.6

def delay(x, secs, fb=0.35, mix=0.3):
    d = int(secs * SR); y = x.copy()
    for k in range(1, 5):
        y[d * k:] += x[: len(x) - d * k] * (fb ** k)
    return (1 - mix) * x + mix * y

# ---------- zene ----------
PADS = [[53, 56, 60], [49, 53, 56], [46, 49, 53], [48, 52, 55]]  # Fm Db Bbm C
BASS = [29, 25, 22, 24]                                            # F1 Db1 Bb0 C1
# Saját lead dallam ütemenként: (kezdés nyolcadban, hossz nyolcadban, MIDI)
MELODY = [
    [(0, 1, 77), (1, 1, 80), (2, 2, 79), (4, 1, 77), (5, 1, 75), (6, 2, 77)],
    [(0, 1, 80), (1, 1, 77), (2, 3, 73), (5, 1, 75), (6, 2, 77)],
    [(0, 1, 77), (1, 1, 80), (2, 2, 82), (4, 1, 80), (5, 1, 77), (6, 2, 73)],
    [(0, 3, 76), (3, 1, 77), (4, 2, 79), (6, 2, 76)],
]
BELL_ARP = [0, 1, 2, 1, 0, 2, 1, 2]

SECTIONS = [  # név, ütem, lead, dob, 808
    ("intro", 4, True, False, False),
    ("hook", 8, True, True, True),
    ("verse", 16, False, True, True),
    ("hook", 8, True, True, True),
    ("verse", 16, False, True, True),
    ("hook", 8, True, True, True),
    ("outro", 4, True, False, False),
]
total_bars = sum(s[1] for s in SECTIONS)
L = int((total_bars * BAR + 3) * SR)
leadb, padb, bellb, drums, bass = (np.zeros(L) for _ in range(5))

def put(buf, sig, t, g=1.0):
    i = int(t * SR); j = min(len(buf), i + len(sig))
    buf[i:j] += sig[: j - i] * g

K, S, H, HO = kick(), snare(), hat(), hat(True)
E = BEAT / 2
bi = 0
for name, bars, lead_on, d_on, b_on in SECTIONS:
    for b in range(bars):
        t0 = bi * BAR; ci = b % 4
        put(padb, pad(PADS[ci], BAR + 0.3), t0, 1.0)
        for s, idx in enumerate(BELL_ARP):
            put(bellb, bell(PADS[ci][idx] + 24), t0 + s * E, 0.5 if s % 2 else 0.7)
        if lead_on:
            for st, ln, n in MELODY[ci]:
                put(leadb, lead(n, ln * E * 0.95), t0 + st * E, 1.0)
        if d_on:
            kp = [0, 1.5, 2.75] if b % 2 == 0 else [0, 0.75, 1.5, 3.25]
            for p in kp:
                put(drums, K, t0 + p * BEAT, 0.9)
            put(drums, S, t0 + 2 * BEAT, 0.85)
            pos, step = 0.0, 0
            roll = b % 4 == 3 or (name == "hook" and b % 2 == 1)
            while pos < 4:
                if roll and 1.0 <= pos < 1.5:   sub = 1 / 6
                elif roll and 3.0 <= pos < 4:   sub = 1 / 8 if pos >= 3.5 else 1 / 4
                elif b % 8 == 5 and pos >= 2:   sub = 1 / 3
                else:                           sub = 0.5
                put(drums, H, t0 + pos * BEAT, 0.5 + 0.3 * (step % 2 == 0))
                pos += sub; step += 1
            if b % 2 == 0:
                put(drums, HO, t0 + 3.5 * BEAT, 0.3)
        if b_on:
            r = hz(BASS[ci] + 12)
            pat = [(0, 1.5), (1.5, 1.25), (2.75, 1.25)] if b % 2 == 0 else \
                  [(0, 0.75), (0.75, 0.75), (1.5, 1.75), (3.25, 0.75)]
            for k, (p, d) in enumerate(pat):
                f, gl = r, None
                if k == len(pat) - 1 and b % 2 == 1:
                    f, gl = r * 2, r        # oktáv csúszás, bouncy
                put(bass, make_808(f, d * BEAT, gl), t0 + p * BEAT, 0.9)
        bi += 1

# intro/outro: tompítva
for a, z in ((0, 4), (total_bars - 4, total_bars)):
    i, j = int(a * BAR * SR), int(z * BAR * SR)
    for buf in (leadb, padb, bellb):
        buf[i:j] = lowpass(buf[i:j], 0.08) * 1.4

leadb = reverb(delay(leadb, BEAT * 0.75), 2.8, 0.4)
bellb = reverb(bellb, 2.0, 0.45)
padb = reverb(padb, 3.0, 0.5)

def norm(x): return x / (np.max(np.abs(x)) + 1e-9)
mus = 0.45 * norm(leadb) + 0.35 * norm(padb) + 0.25 * norm(bellb)
mix = np.tanh(1.1 * (0.75 * mus + 0.55 * norm(drums) + 0.75 * norm(bass)))
fo = int(3 * SR); mix[-fo:] *= np.linspace(1, 0, fo)

d = int(0.014 * SR)
w = 0.12 * np.roll(norm(bellb) + norm(padb), d)
st = np.stack([mix + w, mix - w], axis=1)
st /= np.max(np.abs(st)) / 0.95

here = os.path.dirname(os.path.abspath(__file__))
wav = os.path.join(here, "dark_bounce_155bpm.wav")
with wave.open(wav, "wb") as f:
    f.setnchannels(2); f.setsampwidth(2); f.setframerate(SR)
    f.writeframes((st * 32767).astype("<i2").tobytes())
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-b:a", "256k",
                wav.replace(".wav", ".mp3")], check=True)
os.remove(wav)
print("kész:", wav.replace(".wav", ".mp3"), f"{len(mix)/SR:.1f}s")
