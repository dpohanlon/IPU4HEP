from seaborn import apionly as sns

import matplotlib.pyplot as plt

from matplotlib import rcParams
import matplotlib as mpl
mpl.use('Agg')

plt.style.use(['seaborn-whitegrid', 'seaborn-ticks'])
import matplotlib.ticker as plticker
rcParams['figure.figsize'] = 12, 8
rcParams['axes.facecolor'] = 'FFFFFF'
rcParams['savefig.facecolor'] = 'FFFFFF'
rcParams['figure.facecolor'] = 'FFFFFF'

rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

rcParams['mathtext.fontset'] = 'cm'
rcParams['mathtext.rm'] = 'serif'

rcParams.update({'figure.autolayout': True})

import matplotlib.ticker as plticker

import numpy as np
np.random.seed(42)

from numpy.linalg import inv

from pprint import pprint

import time

from genKFTracks2d import genTracks, xRange, yRange

# Track hits are [x, y]
# Track states are [x, theta, y, phi]

d = 1.0 # Distance between planes
sigma = 10E-2 # Resolution f planes
N = 5 # Number of planes
z = 0.1 # Thickness of absorber
x0 = 0.01 # Radiation length of absorber
theta0 = 10E-3 # Multiple scattering uncertainty (TODO: use formula)

F = np.array( [[1, d, 0, 0], [0, 1, 0, 0], [0, 0, 1, d], [0, 0, 0, 1]] )
G = np.array( [[1/sigma ** 2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1/sigma ** 2, 0], [0, 0, 0, 0]] )
H = np.array( [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]] )

Q = np.zeros(4)

C0 = np.array( [[sigma ** 2, 0, 0, 0], [0, np.pi, 0, 0], [0, 0, sigma ** 2, 0], [0, 0, 0, np.pi]] )

def plotHits(N, plotTracks, digiHits, planeRange, name, ax = None):

    for i in range(1, N + 1):
        ax.plot([i, i], [-3, 3], color = 'k', linestyle = '--', alpha = 0.15)

    ax.plot(np.array(range(1, N + 1)), plotTracks.T, alpha = 0.75)
    ax.plot(np.array(range(1, N + 1)), digiHits, 'x', color = 'k')

    ax.set_ylim(planeRange[0] - 0.1, planeRange[1] + 0.1)
    ax.set_xlim(-0.25, N + 0.25)

    ax.text(0.25, 0.80, "$" + name + "$", fontsize = 18)

    ax.set_ylabel(r'$x$' if name == 'y' else r'$y$', fontsize = 16)
    ax.set_xlabel(r'$z$', fontsize = 16)

    loc = plticker.MultipleLocator(base=0.5) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)

def plotHits2d(N, plotTracks, digiHits, planeRangeX, planeRangeY, ax = None):

    # Plot this first as a hack to get the track colour consistent between plots

    ax.plot(smoothedTrack[:,0,:], smoothedTrack[:,2,:], lw = 0.5)

    for p in range(N):

        # First plane
        xHits = digiHits.T[0,p,:]
        yHits = digiHits.T[1,p,:]

        ax.plot(xHits, yHits, 'x', alpha = 0.2)
        ax.set_xlim(planeRangeX[0] - 0.1, planeRangeX[1] + 0.1)
        ax.set_ylim(planeRangeY[0] - 0.1, planeRangeY[1] + 0.1)

    ax.text(-0.90, 0.80, r"$z$", fontsize = 18)

    ax.set_ylabel(r'$x$', fontsize = 16)
    ax.set_xlabel(r'$y$', fontsize = 16)

    loc = plticker.MultipleLocator(base=0.5)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)

def propagateStateEKF(x):

    x_new = x.copy()
    x_new[0,:] = x[0,:] + d * np.tan(x[1,:])
    x_new[2,:] = x[2,:] + d * np.tan(x[3,:])

    return x_new.T

def jacobianF(x):

    # No batch 'eye'
    f = np.zeros( (x.shape[1], x.shape[0], x.shape[0]) )

    f[:,0,0] = 1
    f[:,1,1] = 1
    f[:,2,2] = 1
    f[:,3,3] = 1

    f[:,0,1] = 1./ (np.cos(x[1,:]) + 1E-10) ** 2
    f[:,2,3] = 1./ (np.cos(x[3,:]) + 1E-10) ** 2

    return f

def projectEKF(F, p, C, Q):

    p_proj = propagateStateEKF(p)

    jF = jacobianF(p)
    C_proj = (jF @ C.T) @ np.transpose(jF, (0, 2, 1)) + Q

    return p_proj, C_proj

def project(F, p, C, Q):

    p_proj = np.einsum('ji,iB->Bj', F, p)

    C_proj = (F @ C).T @ F.T + Q

    return p_proj, C_proj

def filter(p_proj, C_proj, H, G, m):

    HG = H.T @ G

    # Innermost two axes must be 'matrix'
    inv_C_proj = inv(C_proj)

    C = inv(inv_C_proj + HG @ H)

    p = np.einsum('Bij,Bj->Bi', inv_C_proj, p_proj) + np.einsum('ji,iB->Bj', HG, m)
    p = np.einsum('Bij,Bj->Bi', C, p)

    return p, C

def bkgTransport(C, F, C_proj):

    #  Extra transpose (both) to make this work with axis ordering

    return C @ F.T @ inv(C_proj)

def smooth(p_k1_smooth, p_k1_proj, C_k1_smooth, C_k1_proj, p_filtered, C_filtered, A):

    # Also reversed batches!
    p_smooth = p_filtered + np.einsum('Bij,jB->iB', A, p_k1_smooth - p_k1_proj)

    # Transpose only inner 'matrix' dimensions
    C_smooth = C_filtered + A @ (C_k1_smooth - C_k1_proj) @ np.transpose(A, (0, 2, 1))

    return p_smooth, C_smooth

def residual(hits, p_filtered, H):

    return hits - (H @ p_filtered)

def chiSquared(residual, G, C_proj, p_proj, p_filt):

    t1 = residual.T @ G @ residual

    p_diff = p_filt - p_proj
    t2 = p_diff.T @ inv(C_proj) @ p_diff

    return t1 + t2

if __name__ == '__main__':

    nGen = 7
    hits, trueTracks = genTracks(nGen = nGen)

    m0 = np.zeros((4, nGen))
    m0[0,:] = hits[:,0,0] # First plane, x hits
    m0[2,:] = hits[:,0,1] # First plane, y hits

    p0 = m0

    C0 = np.stack([C0 for i in range(nGen)], -1)

    start = time.perf_counter()

    # Batch dim second for p
    # p_proj, C_proj = projectEKF(F, p0, C0, Q)
    p_proj, C_proj = project(F, p0, C0, Q)

    p, C = filter(p_proj, C_proj, H, G, m0)

    # Because batch dims are inconsistent...
    p = p.T
    C = np.transpose(C, (1, 2, 0))

    projectedTrack = [p_proj]
    projectedCov = [C_proj]

    filteredTrack = [p]
    filteredCov = [C]

    for i in range(1, N):

        # p_proj, C_proj = projectEKF(F, p, C, Q)
        p_proj, C_proj = project(F, p, C, Q)

        m = np.zeros((4, nGen))
        m[0,:] = hits[:,i,0] # ith plane, x hits
        m[2,:] = hits[:,i,1] # ith plane, y hits

        p, C = filter(p_proj, C_proj, H, G, m)

        p = p.T
        C = np.transpose(C, (1, 2, 0))

        filteredTrack.append(p)
        filteredCov.append(C)

        projectedTrack.append(p_proj)
        projectedCov.append(C_proj)

    smoothedTrack = [None for i in range(N - 1)] + [filteredTrack[-1]]
    smoothedCov = [None for i in range(N - 1)] + [filteredCov[-1]]

    reversedPlaneIndices = list(range(0, N - 1))
    reversedPlaneIndices.reverse()

    for i in reversedPlaneIndices:

        p_k1_proj, C_k1_proj = projectedTrack[i + 1], projectedCov[i + 1]
        p_filtered, C_filtered = filteredTrack[i], filteredCov[i]
        p_k1_smooth, C_k1_smooth = smoothedTrack[i + 1], smoothedCov[i + 1]

        if i == reversedPlaneIndices[0]:
            C_k1_smooth = np.transpose(C_k1_smooth, (2, 0, 1))

        # Need to have 7, 2, 2 shape because of inversion - fix me!!
        A = bkgTransport(np.transpose(C_filtered, (2, 0, 1)), F, C_k1_proj)

        p_smooth, C_smooth = smooth(p_k1_smooth, p_k1_proj.T, C_k1_smooth, C_k1_proj, p_filtered, np.transpose(C_filtered, (2, 0, 1)), A)

        smoothedTrack[i] = p_smooth
        smoothedCov[i] = C_smooth

    smoothedTrack = np.array(smoothedTrack)
    filteredTrack = np.array(filteredTrack)

    end = time.perf_counter()

    print("Elapsed time = {:.12f} seconds".format(end - start))

    fig = plt.figure(figsize=(8,8))

    ax = fig.add_subplot(2, 2, 1)
    plotHits(N, smoothedTrack[:,0,:].T, hits.T[0,:,:], xRange, 'x', ax = ax)

    ax = fig.add_subplot(2, 2, 2)
    plotHits(N, smoothedTrack[:,2,:].T, hits.T[1,:,:], xRange, 'y', ax = ax)

    ax = fig.add_subplot(2, 2, 3)
    plotHits2d(N, smoothedTrack, hits, xRange, yRange, ax = ax)

    plt.savefig('kfTracks.pdf')
    plt.clf()
