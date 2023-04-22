import EXOSIMS.OpticalSystem.Nemati_2019 as n19
import json

with open("../inputs/template_nemati.json", 'rb') as f: 
    specs = json.loads(f.read())
system = n19.Nemati_2019(k_samp=0.15, Nlensl=5, lam_d=500, lam_c=500
                         , MUF_thruput=0.91, ContrastScenario="CGDsignPerf"
                         , **specs)
print(dir(system))

