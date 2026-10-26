"""Deterministic, bounded synthetic soundscapes and honest regression metrics.

No inference or files are changed here. SNR is measured on active target audio,
not inferred from source gains. Activity boundaries are energy estimates, NOT
word alignments. All components receive the same final anti-clipping gain.
"""
import hashlib
import math
import re
import numpy as np
from scipy.signal import fftconvolve, resample_poly

SR = 16000


def sha_audio(x):
    return hashlib.sha256(np.asarray(x, dtype='<f4').tobytes()).hexdigest()


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64)**2))) if len(x) else 0.


def normalize(x, level=.055):
    return np.asarray(x, dtype=np.float32) * (level / max(rms(x), 1e-9))


def activity(x):
    """20-ms energy support with a fixed 5%-of-peak floor; not a VAD oracle."""
    if not len(x):
        return np.zeros(0, dtype=bool)
    energy = np.sqrt(np.convolve(np.asarray(x, dtype=np.float64)**2,
                                np.ones(min(320, len(x)))/min(320, len(x)), mode='same'))
    return energy > max(1e-4, float(energy.max())*.05)


def loop(x, n, offset=0):
    return np.asarray(x)[(np.arange(n)+offset) % len(x)]


def make_bed(backgrounds, n, seed=42):
    """Two independently offset voices, authored music, colored noise and knocks."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)/SR
    parts = []
    for i, bg in enumerate(backgrounds):
        wave = loop(bg, n, int(rng.integers(len(bg))))
        gate = np.ones(n) if i == 0 else .15+.85*(np.sin(2*np.pi*.43*t+.3)>-.2)
        parts.append(wave*gate/(1+i*.4))
    speech = sum(parts) if parts else np.zeros(n)
    music = sum(np.sin(2*np.pi*f*t+.2*i) for i, f in enumerate([130.813, 164.814, 196., 261.626]))
    music *= .6+.4*np.sin(2*np.pi*.6*t)**2
    noise = np.convolve(rng.normal(size=n+10), np.ones(11)/11, mode='valid')
    knocks = np.zeros(n)
    for start in np.arange(.45, n/SR, 1.7):
        at = int(start*SR); size = min(2400, n-at)
        knocks[at:at+size] += rng.normal(size=size)*np.exp(-np.arange(size)/(SR*.035))
    bed = normalize(speech)*.8 + normalize(music)*.18 + normalize(noise)*.08 + normalize(knocks)*.08
    return normalize(bed), {'background_voices':len(backgrounds), 'procedural_music':True,
                            'colored_noise':True, 'impulsive_noise':True, 'seed':seed}


def soundscape(target, backgrounds, *, seed=42, snr_db=12., onset_s=.64,
               tail_s=2.4, speed=1., reverb_s=0.):
    target = np.asarray(target, dtype=np.float32)
    if speed != 1.:
        target = resample_poly(target, 100, round(100*speed)).astype(np.float32)
    if reverb_s:
        # Sparse, deterministic reflections. This is a synthetic RIR, not a room recording.
        impulse = np.zeros(round(reverb_s*SR)+1); impulse[0] = 1.
        for delay, gain in [(0.023,.22),(.061,.14),(.107,.08),(.173,.04)]:
            if round(delay*SR)<len(impulse):impulse[round(delay*SR)]=gain
        target = fftconvolve(target, impulse).astype(np.float32)
    n = round(onset_s*SR)+len(target)+round(tail_s*SR)
    if n > 120*SR:
        raise ValueError('A soundscape is limited to 120 seconds; stream longer soaks in blocks')
    main = np.zeros(n, np.float32); start = round(onset_s*SR)
    main[start:start+len(target)] = target
    mask = activity(main)
    if not mask.any():raise ValueError('A foreground soundscape needs non-silent target audio')
    bed, meta = make_bed(backgrounds, n, seed)
    if snr_db is None:
        bed *= 0
    else:
        bed *= rms(main[mask]) / max(rms(bed[mask]),1e-9) / (10**(snr_db/20))
    mix = main+bed
    gain = min(1., .94/max(float(np.max(np.abs(mix))),1e-9))
    mix = np.asarray(mix*gain,dtype=np.float32); bed = np.asarray(bed*gain,dtype=np.float32)
    snr = 20*math.log10(rms((main*gain)[mask])/rms(bed[mask])) if rms(bed[mask]) else None
    active = np.flatnonzero(mask)
    return mix, {
        **meta, 'requested_snr_db':snr_db, 'measured_snr_db':snr,
        'snr_definition':'target/interferer RMS on isolated target energy-active samples',
        'target_activity_s':[int(active[0])/SR, (int(active[-1])+1)/SR],
        'target_file_s':[start/SR, (start+len(target))/SR],
        'activity_is_word_alignment':False, 'duration_s':n/SR, 'tail_s':tail_s,
        'speed_ratio':speed, 'synthetic_reverb_s':reverb_s, 'common_peak_gain':gain,
        'peak':float(np.max(np.abs(mix))), 'mixture_sha256':sha_audio(mix),
        'target_sha256':sha_audio(target), 'bed_rms':rms(bed),
    }, bed


def words(text):return re.findall(r'\w+', text.casefold())


def word_error_rate(reference, hypothesis):
    ref, hyp = words(reference), words(hypothesis)
    row = list(range(len(hyp)+1))
    for i, a in enumerate(ref, 1):
        out = [i]
        for j, b in enumerate(hyp, 1):out.append(min(out[-1]+1,row[j]+1,row[j-1]+(a!=b)))
        row = out
    return row[-1]/max(1,len(ref))


def wilson(successes, total):
    if not 0 <= successes <= total:raise ValueError('Invalid numerator/denominator')
    if not total:return None
    z=1.959963984540054; p=successes/total; d=1+z*z/total
    center=(p+z*z/(2*total))/d
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [max(0,center-half),min(1,center+half)]


def zero_event_rate_bound(exposure_seconds):
    """One-sided 95% Poisson upper bound per hour, ONLY when zero events."""
    return -math.log(.05)*3600/exposure_seconds if exposure_seconds>0 else None


def judge_text(text, spec, previous=None):
    checks={'minimum_words':len(words(text)) >= spec.get('min_words',0)}
    for index, pattern in enumerate(spec.get('required',[])):
        checks['required_'+str(index)] = bool(re.search(pattern,text,re.I))
    for index, pattern in enumerate(spec.get('forbidden',[])):
        checks['forbidden_'+str(index)] = not bool(re.search(pattern,text,re.I))
    if spec.get('different_from_previous'):
        checks['not_identical'] = bool(previous) and words(text)!=words(previous)
    return checks
