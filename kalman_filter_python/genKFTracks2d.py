import matplotlib.pyplot as plt

from matplotlib import rcParams
import matplotlib as mpl
mpl.use('Agg')

import seaborn

import time

from pprint import pprint

from scipy.optimize import linear_sum_assignment

plt.style.use(['seaborn-whitegrid', 'seaborn-ticks'])
rcParams['figure.figsize'] = 12, 8
rcParams['axes.facecolor'] = 'FFFFFF'
rcParams['savefig.facecolor'] = 'FFFFFF'
rcParams['figure.facecolor'] = 'FFFFFF'

rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

rcParams['mathtext.fontset'] = 'cm'
rcParams['mathtext.rm'] = 'serif'

rcParams.update({'figure.autolayout': True})

import numpy as np
np.random.seed(42)

d = 1.0 # Distance between planes
sigma = 15E-2 # Resolution of planes
N = 5 # Number of planes
z = 0.1 # Thickness of absorber
x0 = 0.01 # Radiation length of absorber
theta0 = 15E-3 # Multiple scattering uncertainty (TODO: use formula)

initialThetaRange = [-np.arcsin(1/5.), np.arcsin(1/5.)]
initialPhiRange = [-np.arcsin(1/5.), np.arcsin(1/5.)]

xRange = [N * d * np.tan(initialThetaRange[0]), N * d * np.tan(initialThetaRange[1])]
yRange = [N * d * np.tan(initialPhiRange[0]), N * d * np.tan(initialPhiRange[1])]

def plotHits(N, plotTracks, digiHits, planeRange, name, ax = None):

    for i in range(1, N + 1):
        ax.plot([i, i], [-3, 3], color = 'k', linestyle = '--', alpha = 0.15)

    ax.plot(np.array(range(0, N + 1)), plotTracks.T)
    ax.plot(np.array(range(1, N + 1)), digiHits.T, 'x', color = 'k')

    ax.set_ylim(planeRange[0] - 0.1, planeRange[1] + 0.1)
    ax.set_xlim(-0.25, N + 0.25)

def plotHits2d(N, plotTracks, digiHits, planeRangeX, planeRangeY, ax = None):

    # Plot this first as a hack to get the track colour consistent between plots

    ax.plot(plotTracks[:,:,0].T, plotTracks[:,:,1].T)

    for p in range(N):

        # First plane
        xHits = digiHits[:,p,0]
        yHits = digiHits[:,p,1]

        ax.plot(xHits, yHits, 'x')
        ax.set_xlim(planeRangeX[0] - 0.1, planeRangeX[1] + 0.1)
        ax.set_ylim(planeRangeY[0] - 0.1, planeRangeY[1] + 0.1)

def dist(h1, h2):
    return np.linalg.norm(h1 - h2)

def exchangeTrackHits(recoHits, frac = 0.2, prob = 0.2):

    newHits = recoHits.copy()

    # Exchange closest frac hits with probability prob
    # Only exchange hits on the same plane

    # Want to avoid exchanging hits for same tracks?

    for plane in range(N):

        planeHits = newHits[:, plane, :]

        k = int(len(planeHits) * frac)
        select = int(len(planeHits) * frac * prob)

        # Do the naive thing first, revisit if it's too slow

        dists = np.array([[(dist(h1, h2)
                            if not np.all(np.equal(h1, h2)) else np.inf)
                            for h1 in planeHits] for h2 in planeHits])

        # Choose a minimal distance assignment that corresponds to the swaps

        s = linear_sum_assignment(dists)
        s = list(zip(s[0], s[1]))

        # Pick the k nearest hits

        s = sorted(s, key = lambda p : dists[p])
        s = np.array(s[:k])

        # Choose select at random

        sIdx = np.random.choice(range(len(s)), select, replace = False)
        s = list(map(tuple, s[sIdx]))

        # Do the exchange

        planeHits[[x[0] for x in s],:], planeHits[[x[1] for x in s],:] = \
        planeHits[[x[1] for x in s],:], planeHits[[x[0] for x in s],:]

    return newHits

def genTracks(nGen = 10, truthOnly = False, plot = False, exchangeHits = False):

    # Absorber lengths add
    msDists = np.array([i * d for i in range(1, N + 1)])

    # But resulting MS uncertainties add in quadrature
    # TODO: Correct for the actual path length (more oblique tracks see more material)
    msErrors = np.array([np.sqrt(i) * theta0 for i in range(1, N + 1)])

    xBins = np.arange(xRange[0] - 2 * sigma, xRange[1] + 2 * sigma, sigma)
    yBins = np.arange(yRange[0] - 2 * sigma, yRange[1] + 2 * sigma, sigma)

    tanThetas = np.tan(np.random.uniform(*initialThetaRange, nGen))
    tanPhis = np.tan(np.random.uniform(*initialPhiRange, nGen))

    # tanThetas = np.tan([0 for i in range(nGen)])
    xTrueHits = np.outer(tanThetas, d * np.array(range(1, N + 1)))
    yTrueHits = np.outer(tanPhis, d * np.array(range(1, N + 1)))

    trueHits = np.stack((xTrueHits, yTrueHits), -1)

    xPlotTracks = np.hstack((np.zeros((nGen, 1)), xTrueHits)) # Project tracks back to origin @ -1
    yPlotTracks = np.hstack((np.zeros((nGen, 1)), yTrueHits)) # Project tracks back to origin @ -1

    plotTracks = np.stack((xPlotTracks, yPlotTracks), -1)

    # Fix me
    msGauss = np.random.normal(np.zeros(N), msErrors, (nGen, N))

    xHits = xTrueHits + msGauss
    yHits = yTrueHits + msGauss

    xHitMap = np.digitize(xHits, xBins)
    xDigiHits = xBins[xHitMap]

    yHitMap = np.digitize(yHits, yBins)
    yDigiHits = yBins[yHitMap]

    digiHits = np.stack((xDigiHits, yDigiHits), -1)

    if exchangeHits:
        digiHits = exchangeTrackHits(digiHits, frac = 0.35, prob = 0.75)

    if plot:

        fig = plt.figure(figsize=(8,8))

        ax = fig.add_subplot(2, 2, 1)
        plotHits(N, xPlotTracks, xDigiHits, xRange, 'x', ax = ax)

        ax = fig.add_subplot(2, 2, 2)
        plotHits(N, yPlotTracks, yDigiHits, yRange, 'y', ax = ax)

        ax = fig.add_subplot(2, 2, 3)
        plotHits2d(N, plotTracks, digiHits, xRange, yRange, ax = ax)

        plt.savefig('multi.pdf')
        plt.clf()

    if not truthOnly:
        return digiHits, plotTracks
    else:
        return trueHits

if __name__ == '__main__':

    genTracks(nGen = 100, plot = True)
