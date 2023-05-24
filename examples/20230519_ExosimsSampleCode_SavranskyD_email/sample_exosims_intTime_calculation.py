import EXOSIMS.MissionSim
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

# build sim object:
sim = EXOSIMS.MissionSim.MissionSim("sample_exosims_inputspec.json")

# identify targets of interest
hipnums = [32439, 77052, 79672, 26779, 113283]
targnames = [f"HIP {n}" for n in hipnums]
for j, t in enumerate(targnames):
    if t not in sim.TargetList.Name:
        targnames[j] += " A"
        assert targnames[j] in sim.TargetList.Name
sInds = np.array([np.where(sim.TargetList.Name == t)[0][0] for t in targnames])


# assemble information needed for integration time calculation:

# we have only one observing mode defined, so use that
mode = sim.OpticalSystem.observingModes[0]

# use the nominal local zodi and exozodi values
fZ = sim.ZodiacalLight.fZ0
fEZ = sim.ZodiacalLight.fEZ0

# target planet deltaMag (evaluate for a range):
npoints = 100
dMags = np.linspace(20, 25, npoints)

# choose angular separation for coronagraph performance
# this doesn't matter for a flat contrast/throughput, but
# matters a lot when you have real performane curves
# we'll use the default values, which is halfway between IWA/OWA
WA = (mode["OWA"] + mode["IWA"]) / 2


# now we loop through the targets of interest and compute intTimes for each:
intTimes = np.zeros((len(targnames), npoints)) * u.d
for j, sInd in enumerate(sInds):
    intTimes[j] = sim.OpticalSystem.calc_intTime(
        sim.TargetList,
        [sInd] * npoints,
        [fZ.value] * npoints * fZ.unit,
        [fEZ.value] * npoints * fEZ.unit,
        dMags,
        [WA.value] * npoints * WA.unit,
        mode,
    )

plt.figure(1)
plt.clf()
for j in range(len(targnames)):
    plt.semilogy(dMags, intTimes[j], label=targnames[j])

plt.xlabel(rf"Achievable Planet $\Delta$mag @ {WA :.2f}")
plt.ylabel(f"Integration Time ({intTimes.unit})")
plt.legend()
plt.savefig('plot.png')
