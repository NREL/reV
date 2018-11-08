#!/usr/bin/env python
"""Python SAM Simulation Core (PySSC)

Created with SAM version 2017.9.5
"""
import sys
import os
from ctypes import (c_int, c_char_p, c_float, CDLL, c_void_p, pointer,
                    POINTER, byref)


# must be c_double or c_float depending on how defined in sscapi.h
c_number = c_float


class PySSC():
    """Python SAM Simulation Core (PySSC)

    Created with SAM version 2017.9.5
    """
    def __init__(self):
        """Initialize a PySSC object."""
        pyssc_dir = os.path.dirname(os.path.abspath(__file__))
        if sys.platform == 'win32' or sys.platform == 'cygwin':
            self.pdll = CDLL(os.path.join(pyssc_dir, 'ssc.dll'))
        elif sys.platform == 'darwin':
            self.pdll = CDLL(os.path.join(pyssc_dir, 'ssc.dylib'))
        elif sys.platform == 'linux2':
            # instead of relative path, require user to have on LD_LIBRARY_PATH
            self.pdll = CDLL(os.path.join(pyssc_dir, 'ssc.so'))
        else:
            print('Platform not supported ', sys.platform)

        self.INVALID = 0
        self.STRING = 1
        self.NUMBER = 2
        self.ARRAY = 3
        self.MATRIX = 4
        self.INPUT = 1
        self.OUTPUT = 2
        self.INOUT = 3

    @staticmethod
    def b(s):
        """Ensure that name strings are sent to PySSC as bytes type.
        """
        if isinstance(s, str):
            return s.encode()
        elif isinstance(s, bytes):
            return s
        else:
            raise TypeError('Input variable should be a string: '
                            '{} {}'.format(s, type(s)))

    def version(self):
        """version """
        self.pdll.ssc_version.restype = c_int
        return self.pdll.ssc_version()

    def build_info(self):
        """build_info """
        self.pdll.ssc_build_info.restype = c_char_p
        return self.pdll.ssc_build_info()

    def data_create(self):
        """data_create """
        self.pdll.ssc_data_create.restype = c_void_p
        return self.pdll.ssc_data_create()

    def data_free(self, p_data):
        """data_free """
        self.pdll.ssc_data_free(c_void_p(p_data))

    def data_clear(self, p_data):
        """data_clear """
        self.pdll.ssc_data_clear(c_void_p(p_data))

    def data_unassign(self, p_data, name):
        """data_unassign """
        self.pdll.ssc_data_unassign(c_void_p(p_data), c_char_p(name))

    def data_query(self, p_data, name):
        """data_query """
        self.pdll.ssc_data_query.restype = c_int
        return self.pdll.ssc_data_query(c_void_p(p_data), c_char_p(name))

    def data_first(self, p_data):
        """data_first """
        self.pdll.ssc_data_first.restype = c_char_p
        return self.pdll.ssc_data_first(c_void_p(p_data))

    def data_next(self, p_data):
        """data_next """
        self.pdll.ssc_data_next.restype = c_char_p
        return self.pdll.ssc_data_next(c_void_p(p_data))

    def data_set_string(self, p_data, name, value):
        """data_set_string """
        self.pdll.ssc_data_set_string(c_void_p(p_data),
                                      c_char_p(self.b(name)),
                                      c_char_p(self.b(value)))

    def data_set_number(self, p_data, name, value):
        """data_set_number """
        self.pdll.ssc_data_set_number(c_void_p(p_data),
                                      c_char_p(self.b(name)),
                                      c_number(value))

    def data_set_array(self, p_data, name, parr):
        """data_set_array """
        count = len(parr)
        arr = (c_number * count)()
        # set all at once instead of looping
        arr[:] = parr
        return self.pdll.ssc_data_set_array(c_void_p(p_data),
                                            c_char_p(self.b(name)),
                                            pointer(arr), c_int(count))

    def data_set_array_from_csv(self, p_data, name, fn):
        """data_set_array_from_csv """
        f = open(fn, 'rb')
        data = []
        for line in f:
            data.extend([n for n in map(float, line.split(b','))])
        f.close()
        return self.data_set_array(p_data, name, data)

    def data_set_matrix(self, p_data, name, mat):
        """data_set_matrix """
        nrows = len(mat)
        ncols = len(mat[0])
        size = nrows * ncols
        arr = (c_number * size)()
        idx = 0
        for r in range(nrows):
            for c in range(ncols):
                arr[idx] = c_number(mat[r][c])
                idx += 1
        return self.pdll.ssc_data_set_matrix(c_void_p(p_data),
                                             c_char_p(self.b(name)),
                                             pointer(arr), c_int(nrows),
                                             c_int(ncols))

    def data_set_matrix_from_csv(self, p_data, name, fn):
        """data_set_matrix_from_csv """
        f = open(fn, 'rb')
        data = []
        for line in f:
            lst = ([n for n in map(float, line.split(b','))])
            data.append(lst)
        f.close()
        return self.data_set_matrix(p_data, name, data)

    def data_set_table(self, p_data, name, tab):
        """data_set_table """
        return self.pdll.ssc_data_set_table(c_void_p(p_data),
                                            c_char_p(self.b(name)),
                                            c_void_p(tab))

    def data_get_string(self, p_data, name):
        """data_get_string """
        self.pdll.ssc_data_get_string.restype = c_char_p
        return self.pdll.ssc_data_get_string(c_void_p(p_data),
                                             c_char_p(self.b(name)))

    def data_get_number(self, p_data, name):
        """data_get_number """
        val = c_number(0)
        self.pdll.ssc_data_get_number(c_void_p(p_data),
                                      c_char_p(self.b(name)),
                                      byref(val))
        return val.value

    def data_get_array(self, p_data, name):
        """data_get_array """
        count = c_int()
        self.pdll.ssc_data_get_array.restype = POINTER(c_number)
        parr = self.pdll.ssc_data_get_array(c_void_p(p_data),
                                            c_char_p(self.b(name)),
                                            byref(count))
        # extract all at once
        arr = parr[0:count.value]
        return arr

    def data_get_matrix(self, p_data, name):
        """data_get_matrix """
        nrows = c_int()
        ncols = c_int()
        self.pdll.ssc_data_get_matrix.restype = POINTER(c_number)
        parr = self.pdll.ssc_data_get_matrix(c_void_p(p_data),
                                             c_char_p(self.b(name)),
                                             byref(nrows), byref(ncols))
        idx = 0
        mat = []
        for _ in range(nrows.value):
            row = []
            for _ in range(ncols.value):
                row.append(float(parr[idx]))
                idx = idx + 1
            mat.append(row)
        return mat

    # don't call data_free() on the result, it's an internal
    # pointer inside SSC
    def data_get_table(self, p_data, name):
        """data_get_table """
        return self.pdll.ssc_data_get_table(c_void_p(p_data), self.b(name))

    def module_entry(self, index):
        """module_entry """
        self.pdll.ssc_module_entry.restype = c_void_p
        return self.pdll.ssc_module_entry(c_int(index))

    def entry_name(self, p_entry):
        """entry_name """
        self.pdll.ssc_entry_name.restype = c_char_p
        return self.pdll.ssc_entry_name(c_void_p(p_entry))

    def entry_description(self, p_entry):
        """entry_description """
        self.pdll.ssc_entry_description.restype = c_char_p
        return self.pdll.ssc_entry_description(c_void_p(p_entry))

    def entry_version(self, p_entry):
        """entry_version """
        self.pdll.ssc_entry_version.restype = c_int
        return self.pdll.ssc_entry_version(c_void_p(p_entry))

    def module_create(self, name):
        """module_create """
        self.pdll.ssc_module_create.restype = c_void_p
        return self.pdll.ssc_module_create(c_char_p(name))

    def module_free(self, p_mod):
        """module_free """
        self.pdll.ssc_module_free(c_void_p(p_mod))

    def module_var_info(self, p_mod, index):
        """module_var_info """
        self.pdll.ssc_module_var_info.restype = c_void_p
        return self.pdll.ssc_module_var_info(c_void_p(p_mod), c_int(index))

    def info_var_type(self, p_inf):
        """info_var_type """
        return self.pdll.ssc_info_var_type(c_void_p(p_inf))

    def info_data_type(self, p_inf):
        """info_data_type """
        return self.pdll.ssc_info_data_type(c_void_p(p_inf))

    def info_name(self, p_inf):
        """info_name """
        self.pdll.ssc_info_name.restype = c_char_p
        return self.pdll.ssc_info_name(c_void_p(p_inf))

    def info_label(self, p_inf):
        """info_label """
        self.pdll.ssc_info_label.restype = c_char_p
        return self.pdll.ssc_info_label(c_void_p(p_inf))

    def info_units(self, p_inf):
        """info_units """
        self.pdll.ssc_info_units.restype = c_char_p
        return self.pdll.ssc_info_units(c_void_p(p_inf))

    def info_meta(self, p_inf):
        """info_meta """
        self.pdll.ssc_info_meta.restype = c_char_p
        return self.pdll.ssc_info_meta(c_void_p(p_inf))

    def info_group(self, p_inf):
        """info_group """
        self.pdll.ssc_info_group.restype = c_char_p
        return self.pdll.ssc_info_group(c_void_p(p_inf))

    def info_uihint(self, p_inf):
        """info_uihint """
        self.pdll.ssc_info_uihint.restype = c_char_p
        return self.pdll.ssc_info_uihint(c_void_p(p_inf))

    def module_exec(self, p_mod, p_data):
        """module_exec """
        self.pdll.ssc_module_exec.restype = c_int
        return self.pdll.ssc_module_exec(c_void_p(p_mod), c_void_p(p_data))
#        ssc_module_exec_simple_nothread

    def module_exec_simple_no_thread(self, modname, data):
        """module_exec_simple_no_thread """
        self.pdll.ssc_module_exec_simple_nothread.restype = c_char_p
        return self.pdll.ssc_module_exec_simple_nothread(c_char_p(modname),
                                                         c_void_p(data))

    def module_log(self, p_mod, index):
        """module_log """
        log_type = c_int()
        time = c_float()
        self.pdll.ssc_module_log.restype = c_char_p
        return self.pdll.ssc_module_log(c_void_p(p_mod), c_int(index),
                                        byref(log_type), byref(time))

    def module_exec_set_print(self, prn):
        """module_exec_set_print """
        return self.pdll.ssc_module_exec_set_print(c_int(prn))
