Parametrizations for Wilson-Cowan (1973) model
---

Wilson-Cowan (1973) Limit cycle | Sanz-León 2015 limit cycle - ok - 28Hz

    m = models.WilsonCowan(P=np.array([0]), # original P=1.25

                       c_ee=np.array([16]), c_ei=np.array([12]), c_ie=np.array([15]), c_ii=np.array([3]),
                       alpha_e=np.array([1]), alpha_i=np.array([1]),

                       r_e=np.array([1]), r_i=np.array([1]), k_e=np.array([1]), k_i=np.array([1]),
                       tau_e=np.array([8]), tau_i=np.array([8]), theta_e=np.array([0]), theta_i=np.array([0]),

                       a_e=np.array([1.3]), a_i=np.array([2]), b_e=np.array([4]), b_i=np.array([3.7]),
                       c_e=np.array([1]), c_i=np.array([1]))


Daffertshofer 2011 from Sanz-Leon 2014 - ok - 20Hz

    m = models.WilsonCowan(P=np.array([0.5]),

                       c_ee=np.array([10]), c_ei=np.array([6]), c_ie=np.array([10]), c_ii=np.array([1]),
                       alpha_e=np.array([1.2]), alpha_i=np.array([2]),

                       r_e=np.array([0]), r_i=np.array([0]), k_e=np.array([1]), k_i=np.array([1]),
                       tau_e=np.array([10]), tau_i=np.array([10]), theta_e=np.array([2]), theta_i=np.array([3.5]),

                       a_e=np.array([1]), a_i=np.array([1]), b_e=np.array([0]), b_i=np.array([0]),
                       c_e=np.array([1]), c_i=np.array([1]))


Abeysuriya 2018 - ok - 10Hz oscillations

    m = models.WilsonCowan(P=np.array([0.31]), 

                       c_ee=np.array([3.25]), c_ei=np.array([2.5]), c_ie=np.array([3.75]), c_ii=np.array([0]),
                       alpha_e=np.array([1]), alpha_i=np.array([1]),

                       r_e=np.array([0]), r_i=np.array([0]), k_e=np.array([1]), k_i=np.array([1]),
                       tau_e=np.array([10]), tau_i=np.array([20]), theta_e=np.array([0]), theta_i=np.array([0]),

                       a_e=np.array([4]), a_i=np.array([4]), b_e=np.array([1]), b_i=np.array([1]),
                       c_e=np.array([1]), c_i=np.array([1]))


Daffertshofer 2011 - ok - No oscillation

    m = models.WilsonCowan(P=np.array([0.5]),

                       c_ee=np.array([10]), c_ei=np.array([6]), c_ie=np.array([1]), c_ii=np.array([10]),
                       alpha_e=np.array([1.2]), alpha_i=np.array([2]),

                       r_e=np.array([0]), r_i=np.array([0]), k_e=np.array([1]), k_i=np.array([1]),
                       tau_e=np.array([10]), tau_i=np.array([10]), theta_e=np.array([2]), theta_i=np.array([3.5]),

                       a_e=np.array([1]), a_i=np.array([1]), b_e=np.array([0]), b_i=np.array([0]),
                       c_e=np.array([1]), c_i=np.array([1]))

TVB limit cycle - ok - No oscillation
from: http://docs.thevirtualbrain.org/api/tvb.simulator.models.html#module-tvb.simulator.models.wilson_cowan

    m = models.WilsonCowan(P=np.array([0.5]),

                       c_ee=np.array([10]), c_ei=np.array([6]), c_ie=np.array([1]), c_ii=np.array([1]),
                       alpha_e=np.array([1.2]), alpha_i=np.array([2]),

                       r_e=np.array([0]), r_i=np.array([0]), k_e=np.array([1]), k_i=np.array([1]),
                       tau_e=np.array([10]), tau_i=np.array([10]), theta_e=np.array([2]), theta_i=np.array([3.5]),

                       a_e=np.array([1]), a_i=np.array([1]), b_e=np.array([0]), b_i=np.array([0]),
                       c_e=np.array([1]), c_i=np.array([1]))