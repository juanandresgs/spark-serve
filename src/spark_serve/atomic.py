"""Publish a newly prepared directory without replacing concurrent work."""
import ctypes
import os
import sys


def publish_exclusive(stage, output):
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == 'darwin':
        fn = libc.renamex_np
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = fn(os.fsencode(stage), os.fsencode(output), 4)
    elif sys.platform.startswith('linux') and hasattr(libc, 'renameat2'):
        fn = libc.renameat2
        fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        result = fn(-100, os.fsencode(stage), -100, os.fsencode(output), 1)
    else:
        raise RuntimeError('Atomic exclusive directory publication requires Linux or macOS')
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(output))
