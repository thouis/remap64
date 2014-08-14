from khash cimport *
from numpy cimport *

import numpy as np

cimport cython
cimport numpy as cnp

ctypedef fused f_int64:
    int64_t
    uint64_t

cdef class Remapper:
    cdef kh_u64_t *table
    cdef bint packed

    def __cinit__(self):
        self.table = kh_init_u64()
        self.packed = False

    def __dealloc__(self):
        kh_destroy_u64(self.table)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef add_merges(self,
                     f_int64[:] _a,
                     f_int64[:] _b):
        cdef:
            uint64_t a, b
            Py_ssize_t k, n = len(_a), i
            int ret = 0

        assert not self.packed
        assert len(_a) == len(_b)
        with nogil:
            for i in range(n):
                a = <uint64_t> _a[i]
                b = <uint64_t> _b[i]
                a = self.chase(a)
                b = self.chase(b)
                # make a smallest
                if a > b:
                    a, b = b, a
                # if a == b, they've already been merged
                if a != b:
                    # chase will have inserted a and b if they didn't have an entry
                    k = kh_get_u64(self.table, <uint64_t> b)
                    self.table.vals[k] = a
            self.compress_paths()

    cdef uint64_t chase(self, uint64_t val) nogil:
        cdef:
            Py_ssize_t k
            int ret = 0
            uint64_t val_parent
        k = kh_get_u64(self.table, val)
        if k != self.table.n_buckets:
            val_parent = self.table.vals[k]
            if val_parent != val:
                # compress path as we chase
                val_parent = self.chase(val_parent)
                self.table.vals[k] = val_parent
            return val_parent
        else:
            # not in the table, so insert it mapping to itself and return
            k = kh_put_u64(self.table, val, &ret)
            self.table.vals[k] = val
            return val

    cpdef fetch(self):
        cdef:
            Py_ssize_t k, i = 0
            ndarray[uint64_t] result_keys, result_counts
        result_keys = np.empty(self.table.n_occupied, dtype=np.uint64)
        result_counts = np.zeros(self.table.n_occupied, dtype=np.uint64)
        for k in range(self.table.n_buckets):
            if kh_exist(self.table, k):
                result_keys[i] = self.table.keys[k]
                result_counts[i] = self.table.vals[k]
                i += 1
        return result_keys, result_counts

    cdef void compress_paths(self) nogil:
        cdef:
            Py_ssize_t k, i = 0
        with gil:
            assert not self.packed
        for k in range(self.table.n_buckets):
            if kh_exist(self.table, k):
                self.table.vals[k] = self.chase(self.table.vals[k])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef pack(self):
        cdef:
            Py_ssize_t k, i = 0, n
            ndarray[uint64_t] src, dest
            uint64_t[:] s
            uint64_t v, next_val = 0

        assert not self.packed

        src, dest = self.fetch()
        src.sort()
        s = src
        n = len(src)

        with nogil:
            for i in range(n):
                v = s[i]
                k = kh_get_u64(self.table, v)
                if self.table.vals[k] == v:
                    # values maps to self, so give it the next index
                    self.table.vals[k] = next_val
                    next_val += 1
                else:
                    # value maps to a lower index, so map to that index's dest
                    self.table.vals[k] = self.table.vals[kh_get_u64(self.table,
                                                                    self.table.vals[k])]
        self.packed = True

    cpdef remap(self, f_int64[:] a):
        cdef:
           Py_ssize_t k, i = 0

        for i in range(len(a)):
            k = kh_get_u64(self.table, a[i])
            if k != self.table.n_buckets:
                a[i] = self.table.vals[k]
