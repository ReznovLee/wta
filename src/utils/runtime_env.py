import os
import sys
import multiprocessing


def configure_runtime():
    plat = sys.platform
    if plat == 'darwin':
        os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
        os.environ.setdefault('OMP_NUM_THREADS', str(min(4, multiprocessing.cpu_count())))
    elif plat.startswith('win'):
        os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
    else:
        os.environ.setdefault('OMP_NUM_THREADS', str(min(4, multiprocessing.cpu_count())))
    try:
        import torch
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    except Exception:
        pass
