import dace

from dace.config import Config

print('Synchronous debugging enabled:',
      Config.get_bool('compiler', 'cuda', 'syncdebug'))
Config.set('frontend', 'unroll_threshold', value=11)


@dace.program
def myfunction(a: dace.float32, b: dace.float32):
    return a + b


def other_function(c: dace.float32, d: dace.float32):
    return c * d + myfunction(c, d)


dfunc = dace.program(other_function)
myfunction.to_sdfg()

dfunc(dace.float32(32.0), dace.float32(32.0))
