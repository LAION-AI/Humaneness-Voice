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
    if np.count_nonzero(keep)<5:return mic
    alpha=float(np.median(gain[keep]))
    if np.median(abs(gain[keep]-alpha))>max(.003,alpha*.2):return mic
    return mic-alpha*reference


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
