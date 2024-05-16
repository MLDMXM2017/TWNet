"""
the improved peak detection algorithm,three filters by height,distance and width
"""
import numpy as np


def find_peaks(x,height=None,distance=None,width=None):
    """
    the improved peak detection algorithm
    @param x: a 1-D array, a sequence with peaks
    @param height: a number or a 2-element sequence, The first element is always interpreted 
                   the  minimal and the second, if supplied, as the
                   maximal required height.
    @param distance:a number, minimal distance between adjacent peaks 
    @param width:a number or a 2-element sequence, The first element is always interpreted 
                   the  minimal and the second, if supplied, as the
                   maximal required width.
    @return: PL, peaks in 'x' that satisfy all given conditions.
    """
    #form the candidate set M
    x_len=x.shape[0]
    M,L,R=[],[],[]
    i=1,i_max=x_len-1
    while i<i_max:
        if x[i-1]<x[i]:
            ahead=i+1
            while ahead<i_max and x[ahead]==x[i]:
                ahead+=1
            if x[ahead]<x[i]:
                L.append(i)
                R.append(ahead-1)
                t=(i+ahead-1)//2
                M.append(t)
                i=ahead
        i+=1
    #filter by height
    peak_heights=x[M]
    try:
        hmin, hmax = height
    except (TypeError, ValueError):
        hmin, hmax = (height, None)
    h_keep=np.ones(len(M))
    if hmin is not None:
        h_keep &= (hmin <= peak_heights)
    if pmax is not None:
        h_keep &= (peak_heights <= hmax)
    M = M[h_keep]
    
    #filter by distance ,highest priority first -> iterate in reverse order (decreasing)
    d_keep = np.ones(len(M)) 
    priority=x[M]
    priority_to_position = np.argsort(priority)
    for i in range(len(M)-1,-1,-1):
        j = priority_to_position[i]
        if d_keep[j] == 0:
            continue
        k = j - 1
        while 0 <= k and M[j] - M[k] < distance:
            d_keep[k] = 0
            k -= 1

        k = j + 1
        while k < x_len and M[k] - M[j] < distance:
            d_keep[k] = 0
            k += 1
    M=M[d_keep]
    
    #filter by width
    left_bases = np.empty(len(M))
    right_bases = np.empty(len(M))
    for peak_nr in range(len(M)):
        peak = M[peak_nr]
        i_min = 0
        i_max = x_len - 1
        i =left_bases[peak_nr]=peak
        left_min = x[peak]
        while i_min <= i and x[i] <= x[peak]:
            if x[i] < left_min:
                left_min = x[i]
                left_bases[peak_nr] = i
            i -= 1
        i =right_bases[peak_nr]=peak
        right_min = x[peak]
        while i <= i_max and x[i] <= x[peak]:
            if x[i] < right_min:
                right_min = x[i]
                right_bases[peak_nr] = i
            i += 1
    peak_widths=[right_bases[i]-left_bases[i] for i in len(M)]
    try:
        wmin, wmax = width
    except (TypeError, ValueError):
        wmin, wmax = (width, None)
    w_keep=np.ones(len(M))
    if wmin is not None:
        w_keep &= (wmin <= peak_widths)
    if wmax is not None:
        w_keep &= (peak_widths <= wmax)
    M = M[w_keep]
    PL=M
    return PL