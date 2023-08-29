import os
import math
import pytest as pt
import numpy as np
import astropy.units as u
from freak import freak

def test_math():
    num = 25
    assert math.sqrt(num) == 5

def test_default():
    t = freak._demo()
    assert t.int_time[0][0] == 0.505681554577927*u.d

def test_nemati2020():
    t = freak.ErrorBudget(input_dir=os.path.join(".", "test")
                          , ref_json_filename="nemati2020_ref.json"
                          , pp_json_filename="nemati2020_pp.json"
                          , output_json_filename="nemati2020_test_output.json"
                          , contrast_filename="nemati2020_contrast.csv"
                          , target_list=[57443, 15457, 72659]
                          , luminosity=[-0.0737, -0.0669, -0.2572]
                          , eeid=[0.09858, 0.09981, 0.11012]
                          , eepsr=[1.37e-10, 1.35e-10, 2.09e-10]
                          , exo_zodi=3*[3.0])
    num_spatial_modes = 13
    num_temporal_modes = 1
    num_angles = 27
    wfe = (0.96*np.ones((num_temporal_modes, num_spatial_modes)))
    wfsc_factor = 0.5*np.ones_like(wfe)
    sensitivity = (
        np.array(num_angles*[3.21, 4.64, 4.51, 3.78, 5.19, 5.82, 10.6, 8.84
                            , 9.09, 3.68, 9.33, 15.0, 0.745])
                 .reshape(num_angles, num_spatial_modes)
                  )
    t.run_etc(wfe, wfsc_factor, sensitivity)
    print("int_time: {}".format(t.int_time))
    print("ppFact: {}".format(t.ppFact))
    print("working_angles:  {}".format(t.working_angles))
    assert t.int_time[1][1].value*24 == pt.approx(25, 0.5)



