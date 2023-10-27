import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sp

def signal_generator(stype='sin',slen=100,delay=0,phase_o=0,f0=1/20,noise_lvl=0):
    # genereate original signals...
    # stype: keyword to generate the signal : dirac, sin or ricker only
    # slen: number of sample
    # delay (in 'sample'): delay to apply, can be a integer or a float...
    # phase_o (rad): phase offset (or "shift").
    # f0: fundamental frequency for stype="sin" only
    if stype=="dirac":
        s = np.zeros(int(slen))
        if abs(int(delay))>slen-1:
            print("for Dirac, only interger delay between ["+ str(-slen) +","+ str(slen-1)+ "] are considered")
            print("delay is set to 0 in other cases...")
            s[0]=1
        else:
            s[int(delay)]= 1 # we cannot really create a dirac between two samples...
    elif stype=="sin":
        s = np.sin(2*np.pi*f0*(np.arange((int(slen)))-delay))
        if phase_o:
            s = apply_phase_offset(s,phase_o)
    elif stype=="ricker":
        s = sp.ricker(int(slen),5)
        if phase_o:
            s = apply_phase_offset(s,phase_o)
        if delay:
            s = apply_delay_f_domain(s,delay)
    else:
        print('stype should be either "dirac", "sin" or "ricker"')
    return s + (np.random.random((slen,))-0.5)*noise_lvl

def make_plts(a,b,c=None,A=None,B=None,C=None,Amp_thresh=0.6,linr=False):
    # plot the different subplots necessary for the practical
    # (a) the two origional signals
    # (b) the time-domain correlation function. The red dot corresponds to its maximum
    # (c) the amplitude spectra of both original signals, black dots delimitate the most energetic frequencies for the reference signal A. by Energetic we here mean higher than 60% of the max value (empirical choice). 
    # (d) the amplitude cross-spectrum, i.e. the amplitude spetrum of the correlation function
    # (e) the phase spectra corresponding to (c). The Green line correspond to the phase difference np.angle(B)-np.angle(A)
    # (f) the phase spectra corresponding to (d)

    plt.figure()

    
    #(a)
    plt.subplot(231)
    plt.plot(a)
    plt.plot(b)
    plt.title('(a) original signals')
    plt.ylabel('amplitude')
    plt.xlabel('time (sample)')

    if c is not None:
        #(b)
        plt.subplot(234)
        delay_from_xcorr,lag,ind=measure_delay_from_xcorr(c)
        plt.plot(lag,c,'c')
        plt.plot(delay_from_xcorr,c[ind],'ro')
        plt.title('(b) cross-correlation')
        plt.ylabel('amplitude')
        plt.xlabel('time lag (sample)')
        print("delay from t-domain c: %f" %delay_from_xcorr)

    if (A is not None) and (B is not None):
        #(c)
        plt.subplot(232)
        absA = np.abs(A)
        absB = np.abs(B)
        nyquist_ind = int(np.round(len(A)/2))
        plt.plot(absA)
        plt.plot(absB)
        indA = np.where(absA[0:nyquist_ind]>np.max(absA[0:nyquist_ind])*Amp_thresh)[0]
        if len(indA)==1:
            plt.plot(indA,absA[indA],'ko')
        else:
            plt.plot(indA[[-1,0]],absA[indA[[-1,0]]],'ko')
        plt.xlim((0,nyquist_ind))
        plt.title('(c) spectral amplitude')
        plt.ylabel('amplitude')
        plt.xlabel('frequency (sample)')
        
        #(e)
        plt.subplot(233)
        angA = np.angle(A)
        angB = np.angle(B)
        diffang = np.unwrap(angB-angA)
        plt.plot(angA)
        plt.plot(angB)
        plt.plot(diffang)
        plt.xlim((0,nyquist_ind))
        if len(indA)==1:
            plt.plot(indA,diffang[indA],'ko')
        else:
            plt.plot(indA[[-1,0]],diffang[indA[[-1,0]]],'ko')
            if linr:
                delay_from_phase_diff,cst=measure_delay_from_phase(diffang,indA,len(A),linr=True)
                plt.plot(indA,2*np.pi*indA*delay_from_phase_diff/len(A)+cst/len(A),'k--')
                print("delay from f-domain phase difference (linear regression): %f" %delay_from_phase_diff)

        plt.title('(e) phase spectra')
        plt.ylabel('phase (rad)')
        plt.xlabel('frequency (sample)')
        
        delay_from_phase_diff=measure_delay_from_phase(diffang,indA,len(A),linr=False)
        print("delay from f-domain phase difference: %f" %delay_from_phase_diff)

    if C is not None:
        #(c)
        plt.subplot(235)
        absC = np.abs(C)
        nyquist_ind = int(np.round(len(C)/2))
        plt.plot(absC,'c')
        indC = np.where(absC[0:nyquist_ind]>np.max(absC[0:nyquist_ind])*Amp_thresh)[0]
        if len(indC)==1:
            plt.plot(indC,absC[indC],'ko')
        else:
            plt.plot(indC[[-1,0]],absC[indC[[-1,0]]],'ko')
        plt.xlim((0,nyquist_ind))
        plt.title('(d) XS amplitude')
        plt.ylabel('amplitude')
        plt.xlabel('frequency (sample)')

        #(f)
        plt.subplot(236)
        angC = np.unwrap(np.angle(C))
        plt.plot(angC,'c')
        plt.xlim((0,nyquist_ind))
        if len(indC)==1:
            plt.plot(indC,angC[indC],'ko')
        else:
            plt.plot(indC[[-1,0]],diffang[indC[[-1,0]]],'ko')
            if linr:
                delay_from_XS,cst=measure_delay_from_phase(diffang,indC,len(C),linr=True)
                plt.plot(indC,2*np.pi*indC*delay_from_XS/len(C)+cst/len(C),'k--')
                print("delay from f-domain cross-spectrum (linear regression): %f" %delay_from_XS)

        plt.title('(f) phase of XS')
        plt.ylabel('unwrapped phase (rad)')
        plt.xlabel('frequency (sample)')

        delay_from_XS=measure_delay_from_phase(diffang,indC,len(C),linr=False)
        print("delay from f-domain cross-spectrum: %f" %delay_from_XS)

    plt.tight_layout()

def apply_delay_f_domain(s,d=0.):
    # apply a delay d to the input signal s
    d_phase = 2*np.pi*d*np.arange(len(s))/(len(s)) #phase associated to d
    d_phase[int(np.round(len(d_phase)/2))::]-=2*np.pi*d #for the hermitian symmetry of the fft
    return np.real(np.fft.ifft(np.fft.fft(s)*np.exp(-1j*d_phase)))

def apply_phase_offset(s,o=0.):
    # apply a phase "shift" (offse "o") to the input signal "s" 
    offset_array = np.ones(len(s))*o
    offset_array[int(np.round(len(offset_array)/2))::]*=-1
    return np.real(np.fft.ifft(np.fft.fft(s)*np.exp(-1j*offset_array)))

def measure_delay_from_xcorr(c):
    # measure delay dt from the index of the max of c
    lag = np.arange(len(c))+1-np.ceil(len(c)/2)
    ind = np.argmax(c)
    dt = lag[ind]
    return dt,lag,ind

def measure_delay_from_phase(phase,freq,slen,linr=False):
    # compute a delay from the phase spectrum information
    # case 1: a single valid value ("sin") => dt = phi / (2 pi freq)
    # case 2: a range of valid values => dt = D_phi / (2 pi D_freq)
    # case 3: a range of valid values from linear regression (if linr==True)
    # case 4:  phase is empty ... no delay to compute
        if len(freq)==1:
            return phase[freq]*slen/(2*np.pi*freq)
        elif len(freq)>1 and not linr:
            return (phase[freq[-1]]-phase[freq[0]])*slen/(2*np.pi*(freq[-1]-freq[0]))
        elif len(freq)>1 and linr:
            # ....
            print('you have to implement your own linear regression here.')
            print('this function should at least return a delay...')
            # ....
            return 
        else:
            return np.NAN
