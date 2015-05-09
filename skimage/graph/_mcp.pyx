#cython: cdivision=True
#cython: nonecheck=False
"""Cython implementation of Dijkstra's minimum cost path algorithm,
for use with data on a n-dimensional lattice.

Original author: Zachary Pincus
Inspired by code from Almar Klein
Later modifications by Almar Klein (Dec 2013)

License: BSD

Copyright 2009 Zachary Pincus

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import cython
import numpy as np
import heap

cimport numpy as cnp
cimport heap

OFFSET_D = np.int8
OFFSETS_INDEX_D = np.int16
EDGE_D = np.int8
INDEX_D = np.intp
FLOAT_D = np.float64


@cython.boundscheck(False)
@cython.wraparound(False)
def _get_edge_map(shape):
    """Return an array with edge points/lines/planes/hyperplanes marked.

    Given a shape (of length n), return an edge_map array with a shape of
    original_shape + (n,), where, for each dimension, edge_map[...,dim] will
    have zeros at indices not along an edge in that dimension, -1s at indices
    along the lower boundary, and +1s on the upper boundary.

    This allows one to, given an nd index, calculate not only if the index is
    at the edge of the array, but if so, which edge(s) it lies along.

    """
    d = len(shape)
    edges = np.zeros(shape+(d,), order='F', dtype=EDGE_D)
    for i in range(d):
        slices = [slice(None)] * (d+1)
        slices[d] = i
        slices[i] = 0
        edges[tuple(slices)] = -1
        slices[i] = -1
        edges[tuple(slices)] = 1
    return edges


@cython.boundscheck(False)
@cython.wraparound(False)
def _offset_edge_map(shape, offsets):
    """Return an array with positions marked where offsets will step
    out of bounds.

    Given a shape (of length n) and a list of n-d offsets, return a two arrays
    of (n,) + shape: pos_edge_map and neg_edge_map.
    For each dimension xxx_edge_map[dim, ...] has zeros at indices at which
    none of the given offsets (in that dimension) of the given sign (positive
    or negative, respectively) will step out of bounds. If the value is
    nonzero, it gives the largest offset (in terms of absolute value) that
    will step out of bounds in that direction.

    An example will be explanatory:
    >>> offsets = [[-2,0], [1,1], [0,2]]
    >>> pos_edge_map, neg_edge_map = _offset_edge_map((4,4), offsets)
    >>> neg_edge_map[0]
    array([[-1, -1, -1, -1],
          [-2, -2, -2, -2],
          [ 0,  0,  0,  0],
          [ 0,  0,  0,  0]], dtype=int8)

    >>> pos_edge_map[1]
    array([[0, 0, 2, 1],
          [0, 0, 2, 1],
          [0, 0, 2, 1],
          [0, 0, 2, 1]], dtype=int8)

    """
    indices = np.indices(shape)  # indices.shape = (n,)+shape

    #get the distance from each index to the upper or lower edge in each dim
    pos_edges = (shape - indices.T).T
    neg_edges = -1 - indices
    # now set the distances to zero if none of the given offsets could reach
    offsets = np.asarray(offsets)
    maxes = offsets.max(axis=0)
    mins = offsets.min(axis=0)
    for pos, neg, mx, mn in zip(pos_edges, neg_edges, maxes, mins):
        pos[pos > mx] = 0
        neg[neg < mn] = 0
    return pos_edges.astype(EDGE_D), neg_edges.astype(EDGE_D)


@cython.boundscheck(False)
@cython.wraparound(False)
def make_offsets(d, fully_connected):
    """Make a list of offsets from a center point defining a n-dim
    neighborhood.

    Parameters
    ----------
    d : int
        dimension of the offsets to produce
    fully_connected : bool
        whether the neighborhood should be singly- of fully-connected

    Returns
    -------
    offsets : list of tuples of length `d`

    Examples
    --------

    The singly-connected 2-d neighborhood is four offsets:

    >>> make_offsets(2, False)
    [(-1,0), (1,0), (0,-1), (0,1)]

    While the fully-connected 2-d neighborhood is the full cartesian product
    of {-1, 0, 1} (less the origin (0,0)).

    """
    if fully_connected:
        mask = np.ones([3]*d, dtype=np.uint8)
        mask[tuple([1]*d)] = 0
    else:
        mask = np.zeros([3]*d, dtype=np.uint8)
        for i in range(d):
            indices = [1]*d
            indices[i] = (0, -1)
            mask[tuple(indices)] = 1
    offsets = []
    for indices, value in np.ndenumerate(mask):
        if value == 1:
            indices = np.array(indices) - 1
            offsets.append(indices)
    return offsets


@cython.boundscheck(True)
@cython.wraparound(True)
def _unravel_index_fortran(flat_indices, shape):
    """_unravel_index_fortran(flat_indices, shape)

    Given a flat index into an n-d fortran-strided array, return an
    index tuple.

    """
    strides = np.multiply.accumulate([1] + list(shape[:-1]))
    indices = [tuple((idx // strides) % shape) for idx in flat_indices]
    return indices


@cython.boundscheck(True)
@cython.wraparound(True)
def _ravel_index_fortran(indices, shape):
    """_ravel_index_fortran(flat_indices, shape)

    Given an index tuple into an n-d fortran-strided array, return a
    flat index.

    """
    strides = np.multiply.accumulate([1] + list(shape[:-1]))
    flat_indices = [np.sum(strides * idx) for idx in indices]
    return flat_indices


@cython.boundscheck(False)
@cython.wraparound(False)
def _normalize_indices(indices, shape):
    """_normalize_indices(indices, shape)

    Make all indices positive. If an index is out-of-bounds, return None.

    """
    new_indices = []
    for index in indices:
        if len(index) != len(shape):
            return None
        new_index = []
        for i, s in zip(index, shape):
            i = int(i)
            if i < 0:
                i = s + i
            if not (0 <= i < s):
                return None
            new_index.append(i)
        new_indices.append(new_index)
    return new_indices


@cython.boundscheck(True)
@cython.wraparound(True)
def _reverse(arr):
    """Reverse index an array safely, with bounds/wraparound checks on.
    """
    return arr[::-1]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class MCP:
    """MCP(costs, offsets=None, fully_connected=True, sampling=None)

    A class for finding the minimum cost path through a given n-d costs array.

    Given an n-d costs array, this class can be used to find the minimum-cost
    path through that array from any set of points to any other set of points.
    Basic usage is to initialize the class and call find_costs() with a one
    or more starting indices (and an optional list of end indices). After
    that, call traceback() one or more times to find the path from any given
    end-position to the closest starting index. New paths through the same
    costs array can be found by calling find_costs() repeatedly.

    The cost of a path is calculated simply as the sum of the values of the
    `costs` array at each point on the path. The class MCP_Geometric, on the
    other hand, accounts for the fact that diagonal vs. axial moves are of
    different lengths, and weights the path cost accordingly.

    Array elements with infinite or negative costs will simply be ignored, as
    will paths whose cumulative cost overflows to infinite.

    Parameters
    ----------
    costs : ndarray
    offsets : iterable, optional
        A list of offset tuples: each offset specifies a valid move from a
        given n-d position.
        If not provided, offsets corresponding to a singly- or fully-connected
        n-d neighborhood will be constructed with make_offsets(), using the
        `fully_connected` parameter value.
    fully_connected : bool, optional
        If no `offsets` are provided, this determines the connectivity of the
        generated neighborhood. If true, the path may go along diagonals
        between elements of the `costs` array; otherwise only axial moves are
        permitted.
    sampling : tuple, optional
        For each dimension, specifies the distance between two cells/voxels.
        If not given or None, the distance is assumed unit.

    Attributes
    ----------
    offsets : ndarray
        Equivalent to the `offsets` provided to the constructor, or if none
        were so provided, the offsets created for the requested n-d
        neighborhood. These are useful for interpreting the `traceback` array
        returned by the find_costs() method.

    """

    def __init__(self, costs, offsets=None, fully_connected=True,
                 sampling=None):
        """__init__(costs, offsets=None, fully_connected=True, sampling=None)

        See class documentation.
        """
        costs = np.asarray(costs)
        if not np.can_cast(costs.dtype, FLOAT_D):
            raise TypeError('cannot cast costs array to ' + str(FLOAT_D))

        # Check sampling
        if sampling is None:
            sampling = np.array([1.0 for s in costs.shape], FLOAT_D)
        elif isinstance(sampling, (list, tuple)):
            sampling = np.array(sampling, FLOAT_D)
            if sampling.ndim != 1 or len(sampling) != costs.ndim:
                raise ValueError('Need one sampling element per dimension.')
        else:
            raise ValueError('Invalid type for sampling: %r.' % type(sampling))

        # We use flat, fortran-style indexing here (could use C-style,
        # but this is my code and I like fortran-style! Also, it's
        # faster when working with image arrays, which are often
        # already fortran-strided.)
        try:
            self.flat_costs = costs.astype(FLOAT_D, copy=False).ravel('F')
        except TypeError:
            self.flat_costs = costs.astype(FLOAT_D).flatten('F')
            print('Using older Numpy version. Upgrading might decrease memory usage and increase speed.')
        size = self.flat_costs.shape[0]
        self.flat_cumulative_costs = np.empty(size, dtype=FLOAT_D)
        self.dim = len(costs.shape)
        self.costs_shape = costs.shape
        self.costs_heap = heap.FastUpdateBinaryHeap(initial_capacity=128,
                                                    max_reference=size-1)

        # This array stores, for each point, the index into the offset
        # array (see below) that leads to that point from the
        # predecessor point.
        self.traceback_offsets = np.empty(size, dtype=OFFSETS_INDEX_D)

        # The offsets are a list of relative offsets from a central
        # point to each point in the relevant neighborhood. (e.g. (-1,
        # 0) might be a 2d offset).
        # These offsets are raveled to provide flat, 1d offsets that can be
        # used in the same way for flat indices to move to neighboring points.
        if offsets is None:
            offsets = make_offsets(self.dim, fully_connected)
        self.offsets = np.array(offsets, dtype=OFFSET_D)
        self.flat_offsets = np.array(
            _ravel_index_fortran(self.offsets, self.costs_shape),
            dtype=INDEX_D)

        # Instead of unraveling each index during the pathfinding algorithm, we
        # will use a pre-computed "edge map" that specifies for each dimension
        # whether a given index is on a lower or upper boundary (or none at
        # all). Flatten this map to get something that can be indexed as by the
        # same flat indices as elsewhere.
        # The edge map stores more than a boolean "on some edge" flag so as to
        # allow us to examine the non-out-of-bounds neighbors for a given edge
        # point while excluding the neighbors which are outside the array.
        pos, neg = _offset_edge_map(costs.shape, self.offsets)
        self.flat_pos_edge_map = pos.reshape((self.dim, size), order='F')
        self.flat_neg_edge_map = neg.reshape((self.dim, size), order='F')


        # The offset lengths are the distances traveled along each offset
        self.offset_lengths = np.sqrt(np.sum((sampling * self.offsets)**2,
                                      axis=1)).astype(FLOAT_D)
        self.dirty = 0
        self.use_start_cost = 1


    def _reset(self):
        """_reset()
        Clears paths found by find_costs().
        """

        cdef INDEX_T start

        self.costs_heap.reset()
        self.traceback_offsets[...] = -2  # -2 is not reached, -1 is start
        self.flat_cumulative_costs[...] = np.inf
        self.dirty = 0

        # Get starts and ends
        # We do not pass them in as arguments for backwards compat
        starts, ends = self._starts, self._ends

        # push each start point into the heap. Note that we use flat indexing!
        for start in _ravel_index_fortran(starts, self.costs_shape):
            self.traceback_offsets[start] = -1
            if self.use_start_cost:
                self.costs_heap.push_fast(self.flat_costs[start], start)
            else:
                self.costs_heap.push_fast(0, start)


    cdef FLOAT_T _travel_cost(self, FLOAT_T old_cost,
                              FLOAT_T new_cost, FLOAT_T offset_length):
        """ float _travel_cost(float old_cost, float new_cost,
                               float offset_length)
        The travel cost for going from the current node to the next.
        Default is simply the cost of the next node.
        """
        return new_cost


    cpdef int goal_reached(self, INDEX_T index, FLOAT_T cumcost):
        """ int goal_reached(int index, float cumcost)
        This method is called each iteration after popping an index
        from the heap, before examining the neighbours.

        This method can be overloaded to modify the behavior of the MCP
        algorithm. An example might be to stop the algorithm when a
        certain cumulative cost is reached, or when the front is a
        certain distance away from the seed point.

        This method should return 1 if the algorithm should not check
        the current point's neighbours and 2 if the algorithm is now
        done.
        """
        return 0


    cdef void _examine_neighbor(self, INDEX_T index, INDEX_T new_index,
                                FLOAT_T offset_length):
        """ _examine_neighbor(int index, int new_index, float offset_length)
        This method is called once for every pair of neighboring nodes,
        as soon as both nodes become frozen.
        """
        pass


    cdef void _update_node(self, INDEX_T index, INDEX_T new_index,
                           FLOAT_T offset_length):
        """ _update_node(int index, int new_index, float offset_length)
        This method is called when a node is updated.
        """
        pass


    def find_costs(self, starts, ends=None, find_all_ends=True,
                   max_coverage=1.0, max_cumulative_cost=None, max_cost=None):
        """
        Find the minimum-cost path from the given starting points.

        This method finds the minimum-cost path to the specified ending
        indices from any one of the specified starting indices. If no end
        positions are given, then the minimum-cost path to every position in
        the costs array will be found.

        Parameters
        ----------
        starts : iterable
            A list of n-d starting indices (where n is the dimension of the
            `costs` array). The minimum cost path to the closest/cheapest
            starting point will be found.
        ends : iterable, optional
            A list of n-d ending indices.
        find_all_ends : bool, optional
   