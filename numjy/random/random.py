# coding=utf-8

from org.meteothink.math import RandomUtil
from org.meteothink.math.distribution import DistributionUtil
from org.apache.commons.math3.distribution import NormalDistribution, BetaDistribution, \
    ChiSquaredDistribution, ExponentialDistribution

from numjy.core.multiarray import NDArray

__all__ = [
    'chisquare','exponential','normal','rand','randn','randint','poisson','seed'
    ]

def seed(seed=None):
    '''
    Seed the generator.
    
    :param seed: (*int*) Seed for random data generator.
    '''
    if seed is None:
        RandomUtil.useSeed = False
    else:
        RandomUtil.useSeed = True
        RandomUtil.seed = seed
    
def rand(*args):
    """
    Random values in a given shape.
    
    Create an array of the given shape and propagate it with random samples from a uniform 
        distribution over [0, 1).
    
    :param d0, d1, ..., dn: (*int*) optional. The dimensions of the returned array, should all
        be positive. If no argument is given a single Python float is returned.
        
    :returns: Random values array.
    """
    if len(args) == 0:
        return RandomUtil.rand()
    elif len(args) == 1:
        return NDArray(RandomUtil.rand(args[0]))
    else:
        return NDArray(RandomUtil.rand(args))
        
def randn(*args):
    """
    Return a sample (or samples) from the “standard normal” distribution.
    
    Create an array of the given shape and propagate it with random samples from a "normal" 
        (Gaussian) distribution of mean 0 and variance 1.
    
    :param d0, d1, ..., dn: (*int*) optional. The dimensions of the returned array, should all
        be positive. If no argument is given a single Python float is returned.
        
    :returns: Random values array.
    """
    if len(args) == 0:
        return RandomUtil.randn()
    elif len(args) == 1:
        return NDArray(RandomUtil.randn(args[0]))
    else:
        return NDArray(RandomUtil.randn(args))
        
def randint(low, high=None, size=None):
    """
    Return random integers from low (inclusive) to high (exclusive).
    
    Return random integers from the “discrete uniform” distribution of the specified dtype in the “half-open” 
    interval [low, high). If high is None (the default), then results are from [0, low).
    
    :param low: (*int*) Lowest (signed) integer to be drawn from the distribution (unless high=None, in which 
        case this parameter is one above the highest such integer).
    :param high: (*int*) If provided, one above the largest (signed) integer to be drawn from the distribution 
        (see above for behavior if high=None).
    :param size: (*int or tuple*) Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples 
        are drawn. Default is None, in which case a single value is returned.
        
    :returns: (*int or array*) Random integer array.
    """
    if high is None:
        high = low
        low = 0
    else:
        high = high - low
    if size is None:
        r = RandomUtil.randint(high)
        r += low
    else:
        r = NDArray(RandomUtil.randint(high, size))
        if low != 0:
            r += low
    return r
    
def poisson(lam=1.0, size=None):
    """
    Draw samples from a Poisson distribution.
    
    :param lam: (*float*) Expectation of interval, should be >= 0.
    :param size: (*int or tuple*) Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples 
        are drawn. Default is None, in which case a single value is returned.
        
    :returns: (*float or array*) Drawn samples from the parameterized Poisson distribution.
    """
    if size is None:
        r = RandomUtil.poisson(lam)
    else:
        r = NDArray(RandomUtil.poisson(lam, size))
    return r
    
def normal(loc=0.0, scale=1.0, size=None):
    """
    Draw random samples from a normal (Gaussian) distribution.
    
    :param loc: (*float*) Mean (“centre”) of the distribution.
    :param scale: (*float*) Standard deviation (spread or “width”) of the distribution.
    :param size: (*int*) Output shape. If size is None (default), a single value is returned.
    
    """
    dist = NormalDistribution(loc, scale)
    if size is None:
        size = 1
    r = DistributionUtil.rvs(dist, size)
    return NDArray(r)
    
def chisquare(df, size=None):
    """
    Draw samples from a chi-square distribution.
    
    :param df: (*float*) Number of degrees of freedom, should be > 0.
    :param size: (*int*) Output shape. If size is None (default), a single value is returned.
    
    """
    dist = ChiSquaredDistribution(df)
    if size is None:
        size = 1
    r = DistributionUtil.rvs(dist, size)
    return NDArray(r)
    
def exponential(scale=1.0, size=None):
    """
    Draw samples from a exponential distribution.
    
    :param scale: (*float*) The scale parameter.
    :param size: (*int*) Output shape. If size is None (default), a single value is returned.
    
    """
    dist = ExponentialDistribution(scale)
    if size is None:
        size = 1
    r = DistributionUtil.rvs(dist, size)
    return NDArray(r)