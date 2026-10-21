"""Residual acoustic echo delay estimation; no recording or guessed delay."""
import numpy as np
from scipy.signal import correlate,correlation_lags


def suppress_aligned_leakage(mic,reference):
    """Conservative probe-only linear echo residual, not source separation.

    Fit gain ONLY on quiet, strongly correlated 20-ms frames from the existing
    DAC-aligned reference. Without enough such evidence return the original.
    Native TYGER and microphone ownership always retain the untouched signal.
    """
    mic=np.asarray(mic,np.float32);reference=np.asarray(reference,np.float32)
    if len(mic)!=len(reference) or len(mic)<1600:return mic
    n=len(mic)//320;a=mic[:n*320].reshape(n,320);b=reference[:n*320].reshape(n,320)
    aa=np.sum(a*a,axis=1);bb=np.sum(b*b,axis=1);ab=np.sum(a*b,axis=1)
    corr=ab/np.sqrt(np.maximum(aa*bb,1e-15));gain=ab/np.maximum(bb,1e-12)
    keep=(bb>=320*.001**2)&(aa<=max(float(aa.max())*.16,320*.003**2))&(corr>=.8)&(gain>.001)&(gain<=.5)
    if np.count_nonzero(keep)<5:
        # A browser's DAC clock cannot know the room/device's remaining
        # acoustic path. Estimate a bounded residual lag from QUIET samples,
        # then require many independently correlated frames at that lag.
        # This affects only the optional probe, never mic timestamps or TYGER.
        quiet=aa<=max(float(aa.max())*.16,320*.003**2)
        masked=(a*quiet[:,None]).reshape(-1)
        c=correlate(masked,reference[:n*320],mode='full',method='fft')
        lags=correlation_lags(len(masked),n*320)
        eligible=np.flatnonzero(np.abs(lags)<=5600) # measured within ±350 ms
        lag=int(lags[eligible[np.argmax(c[eligible])]])
        shifted=np.zeros_like(reference)
        if lag>0:shifted[lag:]=reference[:-lag]
        elif lag<0:shifted[:lag]=reference[-lag:]
        else:shifted=reference
        trial=shifted[:n*320].reshape(n,320)
        trial_bb=np.sum(trial*trial,axis=1);trial_ab=np.sum(a*trial,axis=1)
        trial_corr=trial_ab/np.sqrt(np.maximum(aa*trial_bb,1e-15))
        trial_gain=trial_ab/np.maximum(trial_bb,1e-12)
        selected=quiet&(trial_bb>=320*.001**2)&(trial_corr>=.8)&(trial_gain>.001)&(trial_gain<=.5)
        frames=np.flatnonzero(selected)
        if len(frames)<8:return mic
        fitted=float(np.median(trial_gain[frames[::2]]));validation=frames[1::2]
        # A single correlation peak is not enough. Check held-out quiet frames
        # before accepting the shifted reference; don't fit on double talk.
        error=np.mean((a[validation]-fitted*trial[validation])**2)
        if error>=float(np.mean(a[validation]**2))*.25:return mic
        reference=shifted;b=trial;bb=trial_bb;ab=trial_ab;gain=trial_gain;keep=selected
    alpha=float(np.median(gain[keep]))
    if np.median(abs(gain[keep]-alpha))>max(.003,alpha*.2):return mic
    residual=mic-alpha*reference
    # DAC resampling and the device path are not necessarily a scalar gain.
    # Fit a tiny FIR ONLY on quiet echo-correlated frames, with separate
    # validation frames. Never fit on the user's louder double-talk interval.
    # This removes phase/colour leakage which otherwise corrupts the embedding
    # of a 200-ms command. No change to native TYGER or captured mic audio.
    frames=np.flatnonzero(keep)
    if len(frames)>=8:
        windows=np.lib.stride_tricks.sliding_window_view(np.pad(reference,(4,4)),9)
        def samples(index):return (index[:,None]*320+np.arange(320)).reshape(-1)
        train=samples(frames[::2]);valid=samples(frames[1::2])
        x=windows[train].astype(np.float64);y=mic[train].astype(np.float64)
        gram=x.T@x;ridge=max(float(np.trace(gram))*1e-4/9,1e-8)
        prior=np.zeros(9);prior[4]=alpha
        try:coeff=np.linalg.solve(gram+ridge*np.eye(9),x.T@y+ridge*prior)
        except np.linalg.LinAlgError:return residual
        old=float(np.mean(residual[valid]**2))
        proposed=mic[valid]-windows[valid]@coeff
        # Bounded filter gain and genuinely improved held-out echo fit.
        if np.sum(abs(coeff))<=max(.02,alpha*4) and float(np.mean(proposed**2))<old*.8:
            residual=(mic-windows@coeff).astype(np.float32)
    return residual


def estimate_echo_delay(mic,reference,sr=16000,max_delay=.45,*,symmetric=False,absolute=False):
    mic=np.asarray(mic,np.float64);reference=np.asarray(reference,np.float64)
    n=min(len(mic),len(reference));mic=mic[:n];reference=reference[:n]
    if n<sr//2 or np.linalg.norm(mic)<1e-5 or np.linalg.norm(reference)<1e-5:
        return {'available':False,'reason':'No measurable reference/echo','confidence':0.}
    corr=correlate(mic,reference,mode='full',method='fft');lags=correlation_lags(n,n)
    keep=(lags>=(-round(max_delay*sr) if symmetric else 0))&(lags<=round(max_delay*sr));indexes=np.flatnonzero(keep)
    index=indexes[np.argmax(np.abs(corr[keep]) if absolute else corr[keep])];lag=int(lags[index])
    a=mic[lag:] if lag>=0 else mic[:n+lag]
    b=reference[:n-lag] if lag>=0 else reference[-lag:]
    confidence=float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-10))
    if absolute:confidence=abs(confidence)
    available=confidence>=.45
    return {'available':available,'delay_ms':lag/sr*1000 if available else None,
        'confidence':confidence,'reason':'Measured acoustic residual' if available else 'Echo cancellation/noise makes delay uncertain'}
