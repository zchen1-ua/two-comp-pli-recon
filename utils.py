import numpy as np
from numba import jit, float64 as f64, int32 as i32

@jit(nopython=True, parallel=True)
def NMSE(f1,f0):
    """return normalize SE between f1 and f0"""
    return np.nansum(np.square(f1-f0))/np.nansum(np.square(f0))

def concat_list_w_nones(list_with_nones):
    """remve None from a list"""
    res = list(filter(None, list_with_nones))
    return np.concate(res,axis=0)

@jit(nopython=True, parallel=True)
def map_bins(data, grid):
    """
    Map data to the index of bins defined by 1D array grid.
    Return array has the same shape as data.
    """
    return np.digitize(data, grid).astype(np.int32)-1

@jit(nopython=True, parallel=True)
def get_theta(x, y, degree=False):
    """
    return angle of (x,y)
    if degree is True, return degree; otherwise, return radian
    """
    theta = np.arctan2(y,x)
    theta[theta<0] += 2*np.pi
    return theta*180./np.pi if (degree) else theta

@jit(nopython=True, parallel=True)
def get_grid(n, dx, center=True):
    """
    compute boundries of n bins, with width = dx
    center = True: shift to put the coordinate origin at center
    return:
        edges (of bins), centers (of bins)
    """
    if (n<=0) or (dx<=0):
        raise ValueError("n or binsize is not positive")
    # get bin center
    centers = (np.arange(0,n)+0.5)*dx
    if (center): centers -= 0.5*(n*dx)
    edges = np.append(centers-0.5*dx, centers[-1]+0.5*dx)
    return edges, centers

@jit(nopython=True, parallel=True)
def intersect_circle(x, y, angle, D):
    """
    find intersections of lines with a circle
    
    Inputs
    -------
    (x,y): a point on the line
    angle: angle of the line; can be an iterable for multiple angles
    D: diameter of the circle

    Returns
    -------
    a1, a2: signed distance from (x,y) along angle to cricle
    theta1, theta2: angles of the intersections on circle
    """
    ux, uy = np.cos(angle), np.sin(angle)
    rdot = x*ux+y*uy
    tt = D**2/4-x**2-y**2
    a1 = -rdot+np.sqrt(rdot**2+tt)
    a2 = -rdot-np.sqrt(rdot**2+tt)
    theta1 = get_theta(x+a1*ux, y+a1*uy)
    theta2 = get_theta(x+a2*ux, y+a2*uy)
    return (a1, a2, theta1, theta2)

def length_in_box(x,y,angle,left,right,bottom,top):
    """
    calculate length of the ray emitting at (x,y) in direction
    angle with the box left<=x<=right, bottom<=y<=top
    (x,y) is assumed to be inside the box
    
    angle can be an iterable for multple angles
    """
    # if angle is a number convert to numpy array of 1 number
    a = np.array([angle]).reshape(-1)
    ux, uy, w = np.cos(a), np.sin(a), np.zeros_like(a)
    # make cosine nonnegative
    s = (ux<0)
    ux[s], uy[s] = -ux[s], -uy[s]
    uxuy = ux*uy
    # calculate distances
    dy1, dy2 = (top-y)*ux, (y-bottom)*ux
    dx1, dx2 = (right-x)*uy, (x-left)*uy
    # handle sine >=0
    s = (uy>0)
    w[s] = np.min([dy1[s],dx1[s]],axis=0)+np.min([dy2[s],dx2[s]],axis=0)
    w[s] /= uxuy[s]
    s = (uy<0)
    w[s] = np.min([dy2[s],-dx1[s]],axis=0)+np.min([dy1[s],-dx2[s]],axis=0)
    w[s] /= -uxuy[s]
    # divide abs(sin*cos) if non-zero
    w[ux==0] = top-bottom
    w[uy==0] = right-left
    return w if isinstance(w,(list,tuple,np.ndarray)) else w[0]

@jit(nopython=True, parallel=True)
def erf(x):
    """
    custom error function because numba
    does not support scipy.special.erf
    """
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    t = 1.0/(1.0 + p*np.abs(x))
    y = 1-(((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x**2)
    return np.sign(x)*y

@jit(nopython=True, parallel=True)
def GaussIntegralPoints(locs, sigma, edges, wts):
    """
    Calculate, for each loc in locs and wt in wts
        wt * 1/(sqrt(2\pi)*sigma) \int exp(-(x'-loc)**2/(2*sigma**2)) dx' 
    over a set of bins
    
    Input:
        locs:
            locations of the Gaussian
        sigma:
            sigma of the Gaussian
        edges:
            edges of the bins to calculate the integral
        wts:
            weights associated with locs
    Return: 
        ndarray of integrals, shape = (# locs, # edges-1)
    """
    re = np.zeros((np.shape(locs)[0], np.shape(edges)[0]-1), dtype=np.float64)
    z = edges/np.sqrt(2)/sigma
    tt = locs/np.sqrt(2)/sigma
    for i in np.arange(np.shape(tt)[0]):
        zz = z-tt[i]
        re[i,:] = 0.5*wts[i]*(erf(zz[1:])-erf(zz[:-1]))
    return re

@jit(f64[:,:](f64[:],f64,f64[:],i32),
     nopython=True,
     parallel=True)
def GaussIntegralGrid(grid, std, edges, nsubsamples):
    """
    grid: edges of a grid
    nsubsamples = 1:
        calculate GaussIntegral(locs,std, edges), locs are midpoints of grid
    otherwise:
        for each bin of grid, calculate GaussIntegral(locs,std, edges),
        locs are nsamples+1 regularly spaced points
    """
    if (nsubsamples==1):
        x = 0.5*(grid[1:]+grid[:-1])
        w = grid[1:]-grid[:-1]
        return GaussIntegralPoints(x, std, edges, w)
    else:
        re = np.zeros((np.shape(grid)[0]-1,np.shape(edges)[0]-1), dtype=np.float64)
        w = np.ones(nsubsamples+1, dtype=np.float64)
        w[0] = 0.5
        w[nsubsamples] = 0.5
        for i in np.arange(np.shape(grid)[0]-1):
            dt = (grid[i+1]-grid[i])/nsubsamples
            x = grid[i] + np.arange(nsubsamples+1)*dt
            re[i,:] = np.sum(GaussIntegralPoints(x, std, edges, w*dt), axis=0)
        return re

def Siddon_pp(p1, p2, xedges, yedges, \
              getwts=None, args=(), interior=False):
    """
    find intersections of a line with a grid, Siddon's algorithm
    
    Inputs
    -------
    p1=(x1,y1), p2=(x2,y2): two points on the line
    xedges, yedges: x and y edges defining the grid
    getwts: a function return weight(s) for each intersecting pixels
        * getwts = None:
            Returns a weight for each intersecting pixel, equal to
            the length of the line inside the pixel.
        * otherwise, must be a callable:
            It shall receive edges of the intersecting pixels
                using the t coordinate on the line, and
                additional arguments given by args.
            It returns contribution(s) of each intersecting pixel.
            Caller decides how to interpret the resulting contribution(s).
    interior = True/False: if p1 or p2 can be inside the grid

    Returns
    -------
    ixs, iys:
        indices of intersecting pixels
    wts: 
        weights of each intersecting pixels
    ts: increasing with min=ts[0]=-0.5L and max=ts[-1]=0.5L, L=distance(p1,p2)
        p1+(ts[1:-1]+0.5)*(p2-p1)/L gives intersections with the grid
        So, t=0 gives midpoint of p1 and p2.
    """
    (x1,y1), (x2,y2)= p1, p2
    ## compute solution of x2= x1 + tx*(xgrid-x1)
    ## give intersections of the line (x1,y1)-(x2,y2)
    ## with the x=xgrid line
    txs = np.array([]) if (x1==x2) else (xedges-x1)/(x2-x1)
    
    ## compute solution of y2= y1 + ty*(ygrid-y1)
    ## give intersections of the line (x1,y1)-(x2,y2)
    ## with the y=ygrid line
    tys = np.array([]) if (y1==y2) else (yedges-y1)/(y2-y1)
    
    ## merge tx & ty sets, also add p1 (t=0) & p2 (t=1)
    ts = np.unique(np.concatenate((txs,tys,np.array([0,1])), axis=0))
    if not interior:
        ## allow intersections between (x1,y1) and (x2,y2) only
        rm_loc = np.nonzero(np.logical_or(ts<0, ts>1))[0]
        ts = np.delete(ts, rm_loc)
    ## sort it so that the intersections go from (x1,y1) to (x2,y2)
    ts = np.sort(ts)
    
    ## determine coordinates of intersections with grid
    xs = x1+ts*(x2-x1)
    ys = y1+ts*(y2-y1)
    ## remove points outside of the grid
    out_x = np.logical_or(xs>np.max(xedges), xs<np.min(xedges))
    out_y = np.logical_or(ys>np.max(yedges), ys<np.min(yedges))
    rm_loc = np.nonzero(np.logical_or(out_x, out_y))[0]
    ts = np.delete(ts, rm_loc)
    xs = np.delete(xs, rm_loc)
    ys = np.delete(ys, rm_loc)

    ## convert ts from [0,1] to [L/2,-L/2]
    ## L is the length between p1 to p2
    L = np.sqrt((x2-x1)**2+(y2-y1)**2)
    ts = (ts-0.5)*L
    
    ## identify indices of intersectinig pixels
    if np.shape(xs)[0]>=2:
        ixs = map_bins((xs[1:]+xs[:-1])/2, xedges)
        iys = map_bins((ys[1:]+ys[:-1])/2, yedges)
        wts = ts[1:]-ts[:-1] if (getwts==None) else getwts(ts,*args)
    else:
        ixs = np.array([])
        iys = np.array([])
        wts = np.array([])
    return (ixs, iys, wts, ts)

def Siddon_pd(p, angle, xedges, yedges):
    """
    find intersections of lines with a grid
    
    Inputs
    -------
    p: a point on the line, can be inside the grid
    angle: angle of the line, maybe an iterable for multiple lines
    xedges, yedges: x and y edges defining the grid
    
    Returns
    -------
    ixs, iys: indices of intersecting pixels in the grid
    lens: intersecting lengths
    ts: p + t*(cos(a),sin(a)) give intersection with the grid
    
    if angles is an iterable, the return is an iterable of elements
    (ixs, iys, lens, ts)
    """
    try: # assume angles is an iterable
        return [Siddon_pp(p, (p[0]+np.cos(a), p[1]+np.sin(a)),\
                    xedges, yedges, getwts=None, interior=True) for a in angle]
    except: #angles is not an iterable
        return Siddon_pp(p, (p[0])+np.cos(angle), p[1]+np.sin(angle),\
                    xedges, yedges, getwts=None, interior=True)

def SiddonTOF_pp(p1, p2, xedges, yedges, TOFstd, TOFedges, \
                 nsubsamples=1, interior=True):
    from constants import c
    TOFstd_t = c*TOFstd/2
    TOFedges_t = c*TOFedges/2
    # swap p1 and p2 because TOF is computed by t2-t1
    return Siddon_pp(p2, p1, xedges, yedges, \
            getwts=GaussIntegralGrid, args=(TOFstd_t,TOFedges_t,nsubsamples), \
            interior=True)