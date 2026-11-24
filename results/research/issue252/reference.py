"""Independent finite-population reasoning; imports no training module."""
from itertools import combinations
import numpy as np


def euclidean_midpoint_statistic(coordinates,triple,references):
    """Closed norm identity, not distance-moment code used in production."""
    points=np.asarray(coordinates,dtype=np.float64)
    m,b,c=points[np.asarray(triple,dtype=np.int64)]
    mean=points[np.asarray(references,dtype=np.int64)].mean(0)
    midpoint=(b+c)/2
    return float(np.dot(m-mean,m-mean)-np.dot(midpoint-mean,midpoint-mean))


def direct_statistic(distance_squared,triple,references):
    m,b,c=triple
    terms=[float(distance_squared[a,m])+.25*float(distance_squared[b,c])
           -.5*(float(distance_squared[a,b])+float(distance_squared[a,c])) for a in references]
    if not terms:raise ValueError('empty reference population')
    return sum(terms)/len(terms)


def finite_reference_ensemble(coordinates,triple,population,sample_size):
    return [euclidean_midpoint_statistic(coordinates,triple,subset)
            for subset in combinations(population,sample_size)]
