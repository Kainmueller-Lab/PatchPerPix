import pycuda
import pycuda.compiler


def make_kernel(code, options=None):
    return pycuda.compiler.DynamicSourceModule(code, options=options)


def alloc_zero_array(shape, dtype):
    return pycuda.driver.managed_zeros(
        shape, dtype=dtype, mem_flags=pycuda.driver.mem_attach_flags.GLOBAL)


def sync(context):
    context.synchronize()

def init_cuda():
    from pycuda.autoinit import context
    # import pycuda.driver as cuda

    # # Initialize CUDA
    # cuda.init()

    # # from pycuda.tools import make_default_context
    # global context
    # # context = make_default_context()
    # ndevices = cuda.Device.count()
    # for devn in range(ndevices):
    #     dev = cuda.Device(devn)
    #     try:
    #         context = dev.make_context(cuda.ctx_flags.SCHED_SPIN)
    #         break
    #     except cuda.Error:
    #         pass

    # device = context.get_device()

    # def _finish_up():
    #     global context
    #     context.pop()
    #     context = None

    #     from pycuda.tools import clear_context_caches
    #     clear_context_caches()

    # import atexit
    # atexit.register(_finish_up)
    return context

def get_cuda_stream():
    import pycuda.driver as cuda
    return cuda.Stream()
