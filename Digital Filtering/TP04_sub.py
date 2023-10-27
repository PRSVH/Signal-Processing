import numpy as np
import h5py
from matplotlib import pyplot as plt
from scipy import signal as sp


def signal_generator(stype='sin',slen=100,delay=0,phase_o=0,f0=1/20,noise_lvl=0,N_bxc=10,fc=None,fs=1.):
    # genereate original signals...
    # stype: keyword to generate the signal : dirac, sin or ricker ...
    # slen: number of sample
    # delay (in 'sample'): delay to apply, can be a integer or a float...
    # phase_o (rad): phase offset (or "shift").
    # N_bxc : boxcar width
    # f0: fundamental frequency for stype="sin" and "chirp"
    # fc: cutoff frequency/ies for "lp" "hp", "bp"
    # fs: sampling rate for normalizing fc...
    # 
    if stype=="step":
        s = np.zeros(int(slen))
        s[delay::]= 1 # we cannot really create a dirac between two samples...
    elif stype=="dirac":
        s = np.zeros(int(slen))
        if abs(int(delay))>slen-1:
            print("for Dirac, only interger delay between ["+ str(-slen) +","+ str(slen-1)+ "] are considered")
            print("delay is set to 0 in other cases...")
            s[0]=1
        else:
            s[int(delay)]= 1 # we cannot really create a dirac between two samples...
    elif stype=="boxcar_fwd":
        s = np.zeros(int(slen))
        s[0:N_bxc]=1
        s=np.fft.fftshift(s)
    elif stype=="boxcar_bwd":
        s = np.zeros(int(slen))
        if N_bxc>1:
            s[-(N_bxc-1)::]=1
        s[0]=1
        s=np.fft.fftshift(s)
    elif stype=="boxcar_ctd":
        s = np.zeros(int(slen))
        s[0]=1
        if N_bxc>1:
            if N_bxc//2==1:
                s[-1::]=1
                s[0:2]=1
            else:
                s[0:N_bxc//2]=1
                s[-(N_bxc//2-1)::]=1
        s=np.fft.fftshift(s)
    elif stype=="sin":
        s = np.sin(2*np.pi*f0*(np.arange((int(slen)))-delay))
    elif stype=="ricker":
        s = sp.ricker(int(slen),5)
    elif stype=="chirp":
        s = sp.chirp(np.arange((int(slen))),f0=f0, f1=f0/1000, t1=int(slen))
    elif stype=="lp":
        s = np.zeros(int(slen))
        s[0]=1
        fc_index = int(fc*slen/fs)
        if fc_index>1:
            if fc_index==1:
                s[-1::]=1
                s[0:2]=1
            else:
                s[0:fc_index]=1
                s[-(fc_index-1)::]=1
        s=np.fft.fftshift(s)
        sp.gaussian(100, std=1)
    elif stype=="hp":
        s = np.ones(int(slen))
        s[0]=0
        fc_index = int(fc*slen/fs)
        if fc_index>1:
            if fc_index==1:
                s[-1::]=0
                s[0:2]=0
            else:
                s[0:fc_index]=0
                s[-(fc_index-1)::]=0
        s=np.fft.fftshift(s)
    elif stype=="bp":
        s = np.zeros(int(slen))
        s[0]=1
        if not isinstance(fc, list):
            print('fc should be a list with 2 values fc=[fmin, fmax]...')
            return
        fc_index1 = int(fc[0]*slen/fs)
        fc_index2 = int(fc[1]*slen/fs)
        if fc_index2>1:
            if fc_index2==1:
                s[-1::]=1
                s[0:2]=1
            else:
                s[0:fc_index2]=1
                s[-(fc_index2-1)::]=1
        if fc_index1>1:
            if fc_index1==1:
                s[-1::]=0
                s[0:2]=0
            else:
                s[0:fc_index1]=0
                s[-(fc_index1-1)::]=0        
        s=np.fft.fftshift(s)   
    else:
        print('stype should be either "dirac", "sin" or "ricker"....')
    return s + (np.random.random((slen,))-0.5)*noise_lvl

def make_plts(s,s_o=None,box=None,box_FD=None,print_delay=False,exo=1):
    # plot the different subplots necessary for the practical
    # exo 1 = MA filter section
    # exo 2 = LP, HP, BP filter section
    # (a) the origional signal(s)
    # (b) the amplitude spectrum/a of the orginal signal(s) 
    # (c) the time domain impulse response of the filter
    # (d) The frequency domaini transfer function of the filter
    # (e) the time domain filter output, i.e. filtered signal
    # (f) the Fourier domain filtered signal

    if exo == 1:
        print("sampling rate = 1 day")
        samples_per_unit = 365.25
        units = ['years', 'cycles/year', 'i.e. days']
        freqange = (0,12)
    elif exo == 2:
        samples_per_unit = 10
        print("sampling rate = 1/10 sec")
        units = ['sec', 'Hz', 'i.e. 0.1 sec...']
        freqange = (-0.5,0.5)
    else:
        print("exo should be either 1 (MA filter) of 2 (LP/HP,BP filters)")
        return

    time_a  =np.arange(0,len(s))/samples_per_unit
    freq_v  =np.fft.fftshift(np.fft.fftfreq(len(s),d=1/samples_per_unit))

    fig_h = plt.figure(figsize=(10,5))

    #(a)
    make_subplot(fig_h,time_a,s,y2=s_o,xlabel='time (' +units[0]+ ')',ylabel='amplitude',
        title='(a) original signal(s)',sub=231,alpha2=0.5,color=None,
        color2='orange')
    #(b)
    A      =np.fft.fftshift(np.abs(np.fft.fft(s)))
    A2     = None
    if s_o is not None:
        A2 =np.fft.fftshift(np.abs(np.fft.fft(s_o)))
    make_subplot(fig_h,freq_v,A,y2=A2,xlabel='frequency (' +units[1]+ ')',ylabel='amplitude',
        title='(b) original spectrum/a',sub=234,xlim=freqange,alpha2=0.5,color=None,
        color2='orange')

    #(c to f)
    if (box is not None) or (box_FD is not None):
        if box_FD is None:
            box_FD = np.fft.fftshift(np.abs(np.fft.fft(box)))
        if box is None:
            box = np.fft.fftshift(np.real(np.fft.ifft(np.fft.fftshift(box_FD))))

        t_box   =np.fft.fftshift(np.fft.fftfreq(len(box),d=1/len(box)))
        freq_box=np.fft.fftshift(np.fft.fftfreq(len(box_FD),d=1/samples_per_unit))

        #(c)
        redbox_b=False
        if exo==1:
            redbox_b=True
        make_subplot(fig_h,t_box,box,xlabel='samples (' +units[2]+ ')',ylabel='amplitude',
            title='(c) impulse response',sub=232,color='k',grid=True,redbox=redbox_b)
        #(d)
        redbox_d=False
        if exo==2:
            redbox_d=True
        make_subplot(fig_h,freq_box,box_FD,xlabel='frequency (' +units[1]+ ')',ylabel='amplitude',
            title='(d) spectrum of (c)',sub=235,xlim=freqange,color='k',redbox=redbox_d)
        #(e)
        signal_filt = np.convolve(box,s,mode='same')       
        if s_o is None:
            s_o = s
        make_subplot(fig_h,time_a,signal_filt,y2=s_o,xlabel='time (' +units[0]+ ')',ylabel='amplitude',
            title='(e) (a) convolved w/ box(c)',sub=233,alpha2=0.5,color='yellowgreen',
            color2='orange')
        if print_delay:
            delay_from_xcorr,lag,ind = measure_delay_from_xcorr(np.correlate(signal_filt,s_o,mode='full'))
            print('delay in sample between the two signals in (e):' + str(delay_from_xcorr))
            
        #(f)
        S      =np.fft.fftshift(np.abs(np.fft.fft(signal_filt)))
        make_subplot(fig_h,freq_v,S,y2=A2,xlabel='frequency (' +units[1]+ ')',ylabel='amplitude',
            title='(f) spectrum/a of (e)',sub=236,xlim=freqange,alpha2=0.5,color='yellowgreen',
            color2='orange')

    plt.tight_layout()

def make_plts_recursion(s,a0,b1,npass=1):
    # plot the different subplots necessary for the practical
    # exo 1 = MA filter section
    # exo 2 = LP, HP, BP filter section
    # (a) the origional signal(s)
    # (b) the amplitude spectrum/a of the orginal signal(s) 
    # (c) the time domain impulse response of the filter
    # (d) The frequency domaini transfer function of the filter
    # (e) the time domain filter output, i.e. filtered signal
    # (f) the Fourier domain filtered signal

    samples_per_unit = 10
    print("sampling rate = 1/10 sec")
    units = ['sec', 'Hz', 'i.e. 0.1 sec...']
    freqange = (-0.5,0.5)

    time_a  =np.arange(0,len(s))/samples_per_unit
    freq_v  =np.fft.fftshift(np.fft.fftfreq(len(s),d=1/samples_per_unit))
    fig_h   =plt.figure(figsize=(10,5))

    #(a)
    make_subplot(fig_h,time_a,s,xlabel='time (' +units[0]+ ')',ylabel='amplitude',
        title='(a) original signal(s)',sub=221,alpha2=0.5,color=None,
        color2='orange')
    #(b)
    A      =np.fft.fftshift(np.abs(np.fft.fft(s)))
    make_subplot(fig_h,freq_v,A,xlabel='frequency (' +units[1]+ ')',ylabel='amplitude',
        title='(b) original spectrum/a',sub=223,xlim=freqange,alpha2=0.5,color=None,
        color2='orange')

    #(c)
    signal_filt = my_recursive_filter(s,a0,b1,npass)

    make_subplot(fig_h,time_a,signal_filt,y2=s,xlabel='time (' +units[0]+ ')',ylabel='amplitude',
        title='(e) filtered signal',sub=222,alpha2=0.5,color='yellowgreen',
        color2='orange')
    #(f)
    S_0    =np.fft.fftshift(np.abs(np.fft.fft(s)))
    S      =np.fft.fftshift(np.abs(np.fft.fft(signal_filt)))
    make_subplot(fig_h,freq_v,S,y2=S_0,xlabel='frequency (' +units[1]+ ')',ylabel='amplitude',
        title='(f) spectrum/a of (e)',sub=224,xlim=freqange,alpha2=0.5,color='yellowgreen',
        color2='orange')

    plt.tight_layout()

def make_plts_butter(time,data,data_filt,dirac,dirac_filt,freq,freq2,lowerlim=None,upperlim=None):
    fig_h = plt.figure(figsize=(6,6))
    fig_h.add_subplot(211)
    plt.plot(time,data)
    plt.plot(time,data_filt)
    plt.ylabel('amplitude')
    plt.xlabel('time (sec)')
    plt.title('(a) seismograms')
    plt.legend(['input','output'])
    if (lowerlim !=None and upperlim !=None):
        plt.xlim(lowerlim,upperlim)

    fig_h.add_subplot(223)
    plt.plot(dirac,'k')
    plt.plot(dirac_filt,'r')
    plt.ylabel('amplitude')
    plt.xlabel('time (samples)')
    plt.title('(b) impulse response')
    plt.legend(['input','output'])
    plt.xlim(0,len(dirac_filt)/20)

    fig_h.add_subplot(224)
    S=np.fft.fftshift(np.abs(np.fft.fft(data)))
    S_max=np.max(S)
    S=S/S_max
    S_filt=np.fft.fftshift(np.abs(np.fft.fft(data_filt)))/S_max
    plt.plot(freq,S,alpha=0.5)
    plt.plot(freq,S_filt)
    plt.plot(freq2,np.fft.fftshift(np.abs(np.fft.fft(dirac_filt))),'r')
    plt.xlim(0,0.1)
    plt.ylabel('normalized amplitude')
    plt.xlabel('frequency (Hz)')
    plt.title('(c) frequency responses')

    plt.tight_layout()


def make_subplot(fig_h,x,y,y2=None,xlabel='x',ylabel='y',title='title',
                 sub=111,xlim=None,ylim=None,grid=False,alpha=None,alpha2=0.5,
                 color=None,color2=None,redbox=False):
    ax  = fig_h.add_subplot(sub)
    if y2 is not None:
        plt.plot(x,y2,alpha=alpha2,color=color2)
    plt.plot(x,y,alpha=alpha,color=color)
    if redbox:
        ax.spines['top'].set_color('red')
        ax.spines['bottom'].set_color('red')
        ax.spines['right'].set_color('red')
        ax.spines['left'].set_color('red')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if xlim:plt.xlim(xlim)
    if ylim:plt.xlim(xlim)
    if grid:plt.grid(True)

def measure_delay_from_xcorr(c):
    # measure delay dt from the index of the max of c
    lag = np.arange(len(c))+1-np.ceil(len(c)/2)
    ind = np.argmax(c)
    dt = lag[ind]
    return dt,lag,ind

def read_matv73(filename):
    # input filename (mat, h5 ...) and return a dict of variables
    f=h5py.File(filename,'r')
    dic={}
    for k in f.keys():
        dic[k] = f[k][()][0,:]
    return dic

def my_recursive_filter(s,a0=0.1,b1=0.9,npass=1):
    signalfilt = myrecursive( s, a0, b1 )
    if npass == 2:
        s            = np.flipud(signalfilt)
        signalfilt   = myrecursive( s, a0, b1 )
        signalfilt   = np.flipud(signalfilt)
    return signalfilt

def myrecursive( signal, a0, b1 ):
    #𝑦𝑛 =𝑎0𝑥𝑛 +𝑎1𝑥𝑛−1 +𝑎2𝑥𝑛−2 +𝑎3𝑥𝑛−3 + …+ 𝑏1𝑦𝑛−1 + 𝑏2𝑦𝑛−2+𝑏3𝑦𝑛−3+…
    signalfilt = np.zeros(len(signal))
    signalfilt[0]=a0*signal[0]
    for i in np.arange(1,len(signal)):
        
        signalfilt[i]= a0* signal[i]+b1*signalfilt[i-1]

    return signalfilt
