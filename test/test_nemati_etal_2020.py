import math
from freak import freak

def test_math():
    num = 25
    assert math.sqrt(num) == 5

def test_bogus():
    t = freak._demo()
    assert t.int_time[0][0] == 0.505681554577927


