from tvb.simulator.lab import *
import numpy as np
import glob

ctb_folder = "E:\\LCCN_Local\PycharmProjects\CTB_data2\\"
emp_subj="NEMOS_035"
stimulation = "roast_P3P4Model"
w=0.3
simLength=1000
samplingFreq=1000
g=12
s=12

coup = coupling.SigmoidalJansenRit(a=np.array([g]), cmax=np.array([0.005]), midpoint=np.array([6]),
                                   r=np.array([0.56]))

conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2.zip")

conn.speed = np.array([s])

m = models.JansenRit(A=np.array([3.25]), B=np.array([22]), J=np.array([1]),
                     a=np.array([0.1]), a_1=np.array([135]), a_2=np.array([108]),
                     a_3=np.array([33.75]), a_4=np.array([33.75]), b=np.array([0.06]),
                     mu=np.array([0.1085]), nu_max=np.array([0.0025]), p_max=np.array([0]),
                     p_min=np.array([0]),
                     r=np.array([0.56]), v0=np.array([6]))

weighting = np.loadtxt(glob.glob(
    ctb_folder + 'CurrentPropagationModels/' + emp_subj + '-efnorm_mag-' + stimulation + '*-AAL2.txt')[0],
                       delimiter=",") * w
conn = connectivity.Connectivity.from_file(ctb_folder + emp_subj + "_AAL2.zip")

eqn_t = equations.Noise()
eqn_t.parameters["mean"] = 0
eqn_t.parameters["std"] = (1 - eqn_t.parameters["mean"]) / 3  # p(mean<x<mean+std) = 0.34 in gaussian distribution [max=1; min=-1]
eqn_t.parameters["onset"] = 0
eqn_t.parameters["offset"] = simLength

# eqn_t = equations.Sinusoid()
# eqn_t.parameters['amp'] = 1
# eqn_t.parameters['frequency'] = 10 # Hz
# eqn_t.parameters['onset'] = 0  # ms
# eqn_t.parameters['offset'] = simLength  # ms


stimulus = patterns.StimuliRegion(temporal=eqn_t, connectivity=conn, weight=weighting)

# Configure space and time
stimulus.configure_space()
stimulus.configure_time(np.arange(0, simLength, 1))

plot_pattern(stimulus)



integrator = integrators.HeunDeterministic(dt=1000 / samplingFreq)
mon = (monitors.Raw(),)

sim = simulator.Simulator(model=m, connectivity=conn, coupling=coup, integrator=integrator, monitors=mon,
                          stimulus=stimulus)
sim.configure()
output = sim.run(simulation_length=simLength)