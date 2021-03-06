import numpy as np
import matplotlib

def cbcol(n,withblack=True):
   """give n colors that are color blind friendly

      Example:
        col=nico.cbcol(3,withblack=False)
        plot(x,y0,color=col[0])
        plot(x,y1,color=col[1])
        plot(x,y2,color=col[2])
   """
   if withblack:
     cbbPalette=["#000000", "#E69F00", "#56B4E9", "#CC79A7", "#F0E442", "#0072B2", "#D55E00", "#999999", "#009E73", "#620251"]
   else:
     cbbPalette=["#E69F00", "#56B4E9", "#CC79A7", "#620251", "#F0E442", "#0072B2", "#D55E00", "#999999", "#009E73"]
   ncb=np.size(cbbPalette)
   if n > ncb:
     print('The cbcol fucnction is not defined for more colors than {}'.format(ncb))
     exit()
     #raise Exception('This fucnction is not defined for more colors than {}'.format(ncb))
   col = []
   for kk in np.arange(0,ncb):
     col.append(matplotlib.colors.hex2color(cbbPalette[kk]))
   return col


def sigdigit(a,n):
   """round a to n significant digits

      Examples:
        nico.sigdigit([0.,1.111111,0.],2)          -> array([0. , 1.1, 0. ])
        nico.sigdigit([999.9,1.111111,-323.684],2) -> array([1000. , 1.1, -320. ])
        nico.sigdigit(2.2222222222,3)              -> array([2.22])
        nico.sigdigit(0.,3)                        -> array([0.])
        nico.sigdigit([0.,0.,0.],3)                -> array([0., 0., 0.])

   """
   aa=np.array(a)
   masked = aa==0
   bb=np.ones(np.size(aa))
   if np.size(bb[~masked]) != 0:
     bb[~masked]=np.power(10,np.floor(np.log10(np.abs(aa[~masked]))))
     return np.rint(10**(n-1)*aa/bb)*10**(1-n)*bb
   else:
     return bb*0.e0


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        #raise ValueError, "smooth only accepts 1 dimension arrays."
        print("smooth only accepts 1 dimension arrays.")
        exit()

    if x.size < window_len:
        #raise ValueError, "Input vector needs to be bigger than window size."
        print("Input vector needs to be bigger than window size.")
        exit()

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        #raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        print("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        exit()

    sx = np.size(x)
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    #y=np.convolve(w/w.sum(),s,mode='valid')
    y=np.convolve(w/w.sum(),s,mode='same')
    return y[np.size(x[window_len-1:0:-1]):np.size(x[window_len-1:0:-1])+sx]
