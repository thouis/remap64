from numpy cimport uint64_t, uint32_t, int64_t

cdef extern from "khash.h":
    ctypedef uint32_t khint_t
    ctypedef uint64_t khuint64_t

    ctypedef khint_t khiter_t

    ctypedef struct kh_u64_t:
        khint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        uint64_t *keys
        uint64_t *vals

    inline kh_u64_t* kh_init_u64()
    inline void kh_destroy_u64(kh_u64_t*)
    inline void kh_clear_u64(kh_u64_t*)
    inline khint_t kh_get_u64(kh_u64_t*, uint64_t)
    inline void kh_resize_u64(kh_u64_t*, khint_t)
    inline khint_t kh_put_u64(kh_u64_t*, uint64_t, int*)
    inline void kh_del_u64(kh_u64_t*, khint_t)
    bint kh_exist(kh_u64_t*, khiter_t)
