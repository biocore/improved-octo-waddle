cimport libc.stdlib
from cpython cimport Py_INCREF, Py_DECREF, bool
cimport numpy as np
np.import_array()

cdef class bitarray:
    cdef:
        BIT_ARRAY* bitarr
        bit_index_t nbits   ### REDUNDANT as the BIT_ARRAY struct has this

    def __cinit__(self, bit_index_t n):
        self.nbits = n
        self.bitarr = bit_array_create(n)

    def __dealloc__(self):
        bit_array_free(self.bitarr)

    def __str__(self):
        cdef char* str_
        cdef object result

        str_ = <char*>libc.stdlib.malloc(self.nbits + 1)
        result = tounicode(bit_array_to_str(self.bitarr, str_))
        libc.stdlib.free(str_)

        return result

    def __getitem__(self, bit_index_t i):
        # could inline the macro version as well, less safe but should be 
        # thinner code
        return bit_array_get_bit(self.bitarr, i)

    def __setitem__(self, bit_index_t i, bool v):
        if v:
            bit_array_set_bit(self.bitarr, i)
        else:
            bit_array_clear_bit(self.bitarr, i)

    

cdef unicode tounicode(char* s):
    # from http://docs.cython.org/en/latest/src/tutorial/strings.html
    return s.decode('UTF-8', 'strict')


cpdef bitarray bitarray_factory(object vec):
    """Construct a bitarray from a vector

    A value at a position in the vector is considered set if the value 
    evaluates True.

    Parameters
    ----------
    vec : object
        An iterable

    Returns
    -------
    bitarray
        The bitarray based off of topo
    """
    cdef int i
    cdef int n = len(vec)
    cdef bitarray result
    
    result = bitarray(n)
    for i in range(n):
        if vec[i]:
            result[i] = True

    return result
