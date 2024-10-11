# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)

"""
Jansen-Rit and Wilson-Cowan model combination.
https://groups.google.com/g/tvb-users/c/qQBv8S3nCxc/m/MrVQQTWQBgAJ?hl=es
"""
import math
import numpy
from .base import ModelNumbaDfun, Model
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, List, Range, Final


class JansenRit_WilsonCowan(ModelNumbaDfun):
    r"""
    The Jansen and Rit is a biologically inspired mathematical framework
    originally conceived to simulate the spontaneous electrical activity of
    neuronal assemblies, with a particular focus on alpha activity, for instance,
    as measured by EEG. Later on, it was discovered that in addition to alpha
    activity, this model was also able to simulate evoked potentials.

    .. [JR_1995]  Jansen, B., H. and Rit V., G., *Electroencephalogram and
        visual evoked potential generation in a mathematical model of
        coupled cortical columns*, Biological Cybernetics (73) 357:366, 1995.

    .. [J_1993] Jansen, B., Zouridakis, G. and Brandt, M., *A
        neurophysiologically-based mathematical model of flash visual evoked
        potentials*

    .. figure :: img/JansenRit_45_mode_0_pplane.svg
        :alt: Jansen and Rit phase plane (y4, y5)

        The (:math:`y_4`, :math:`y_5`) phase-plane for the Jansen and Rit model.

    The dynamic equations were taken from [JR_1995]_

    .. math::
        \dot{y_0} &= y_3 \\
        \dot{y_3} &= A a\,S[y_1 - y_2] - 2a\,y_3 - a^2\, y_0 \\
        \dot{y_1} &= y_4\\
        \dot{y_4} &= A a \,[p(t) + \alpha_2 J + S[\alpha_1 J\,y_0]+ c_0]
                    -2a\,y - a^2\,y_1 \\
        \dot{y_2} &= y_5 \\
        \dot{y_5} &= B b (\alpha_4 J\, S[\alpha_3 J \,y_0]) - 2 b\, y_5
                    - b^2\,y_2 \\
        S[v] &= \frac{2\, \nu_{max}}{1 + \exp^{r(v_0 - v)}}


    Wilson-Cowan model -- -- - -- - -- - - - -- -

  **References**:

    .. [WC_1972] Wilson, H.R. and Cowan, J.D. *Excitatory and inhibitory
        interactions in localized populations of model neurons*, Biophysical
        journal, 12: 1-24, 1972.
    .. [WC_1973] Wilson, H.R. and Cowan, J.D  *A Mathematical Theory of the
        Functional Dynamics of Cortical and Thalamic Nervous Tissue*

    .. [D_2011] Daffertshofer, A. and van Wijk, B. *On the influence of
        amplitude on the connectivity between phases*
        Frontiers in Neuroinformatics, July, 2011

    Used Eqns 11 and 12 from [WC_1972]_ in ``dfun``.  P and Q represent external
    inputs, which when exploring the phase portrait of the local model are set
    to constant values. However in the case of a full network, P and Q are the
    entry point to our long range and local couplings, that is, the  activity
    from all other nodes is the external input to the local population.

    The default parameters are taken from figure 4 of [WC_1972]_, pag. 10

    +---------------------------+
    |          Table 0          |
    +--------------+------------+
    |Parameter     |  Value     |
    +==============+============+
    | k_e, k_i     |    0.00    |
    +--------------+------------+
    | r_e, r_i     |    0.00    |
    +--------------+------------+
    | tau_e, tau_i |    9.0    |
    +--------------+------------+
    | c_ee         |    11.0    |
    +--------------+------------+
    | c_ei         |    3.0     |
    +--------------+------------+
    | c_ie         |    12.0    |
    +--------------+------------+
    | c_ii         |    10.0    |
    +--------------+------------+
    | a_e          |    0.2     |
    +--------------+------------+
    | a_i          |    0.0     |
    +--------------+------------+
    | b_e          |    1.8     |
    +--------------+------------+
    | b_i          |    3.0     |
    +--------------+------------+
    | theta_e      |    -1.0     |
    +--------------+------------+
    | theta_i      |    -1.0     |
    +--------------+------------+
    | alpha_e      |    1.0     |
    +--------------+------------+
    | alpha_i      |    1.0     |
    +--------------+------------+
    | P            |    -1.0     |
    +--------------+------------+
    | Q            |    -1.0     |
    +--------------+------------+
    | c_e, c_i     |    0.0     |
    +--------------+------------+
    | shift_sigmoid|    True    |
    +--------------+------------+

    In [WC_1973]_ they present a model of neural tissue on the pial surface is.
    See Fig. 1 in page 58. The following local couplings (lateral interactions)
    occur given a region i and a region j:

      E_i-> E_j
      E_i-> I_j
      I_i-> I_j
      I_i-> E_j


    +---------------------------+
    |          Table 1          |
    +--------------+------------+
    |                           |
    |  SanzLeonetAl,   2014     |
    +--------------+------------+
    |Parameter     |  Value     |
    +==============+============+
    | k_e, k_i     |    1.00    |
    +--------------+------------+
    | r_e, r_i     |    0.00    |
    +--------------+------------+
    | tau_e, tau_i |    10.0    |
    +--------------+------------+
    | c_ee         |    10.0    |
    +--------------+------------+
    | c_ei         |    6.0     |
    +--------------+------------+
    | c_ie         |    10.0    |
    +--------------+------------+
    | c_ii         |    1.0     |
    +--------------+------------+
    | a_e, a_i     |    1.0     |
    +--------------+------------+
    | b_e, b_i     |    0.0     |
    +--------------+------------+
    | theta_e      |    2.0     |
    +--------------+------------+
    | theta_i      |    3.5     |
    +--------------+------------+
    | alpha_e      |    1.2     |
    +--------------+------------+
    | alpha_i      |    2.0     |
    +--------------+------------+
    | P            |    0.5     |
    +--------------+------------+
    | Q            |    0.0     |
    +--------------+------------+
    | c_e, c_i     |    1.0     |
    +--------------+------------+
    | shift_sigmoid|    False   |
    +--------------+------------+
    |                           |
    |  frequency peak at 20  Hz |
    |                           |
    +---------------------------+


    The parameters in Table 1 reproduce Figure A1 in  [D_2011]_
    but set the limit cycle frequency to a sensible value (eg, 20Hz).

    Model bifurcation parameters:
        * :math:`c_1`
        * :math:`P`



    The models (:math:`E`, :math:`I`) phase-plane, including a representation of
    the vector field as well as its nullclines, using default parameters, can be
    seen below:

        .. _phase-plane-WC:
        .. figure :: img/WilsonCowan_01_mode_0_pplane.svg
            :alt: Wilson-Cowan phase plane (E, I)

            The (:math:`E`, :math:`I`) phase-plane for the Wilson-Cowan model.


    The general formulation for the \textit{\textbf{Wilson-Cowan}} model as a
    dynamical unit at a node $k$ in a BNM with $l$ nodes reads:

    .. math::
            \dot{E}_k &= \dfrac{1}{\tau_e} (-E_k  + (k_e - r_e E_k) \mathcal{S}_e (\alpha_e \left( c_{ee} E_k - c_{ei} I_k  + P_k - \theta_e + \mathbf{\Gamma}(E_k, E_j, u_{kj}) + W_{\zeta}\cdot E_j + W_{\zeta}\cdot I_j\right) ))\\
            \dot{I}_k &= \dfrac{1}{\tau_i} (-I_k  + (k_i - r_i I_k) \mathcal{S}_i (\alpha_i \left( c_{ie} E_k - c_{ee} I_k  + Q_k - \theta_i + \mathbf{\Gamma}(E_k, E_j, u_{kj}) + W_{\zeta}\cdot E_j + W_{\zeta}\cdot I_j\right) ))


    """

    ## JANSEN-RIT parameters
    # Define traited attributes for this model, these represent possible kwargs.
    # Define traited attributes for this model, these represent possible kwargs.
    He = NArray(
        label=":math:`He`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain in the first kinetic population.""")

    Hi = NArray(
        label=":math:`B`",
        default=numpy.array([22.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain in the first kinetic population.""")

    tau_e = NArray(
        label=":math:`a`",
        default=numpy.array([10.0]),
        domain=Range(lo=6.3, hi=20.0, step=0.1),
        doc="""Excitatory time constant (ms) in the first kinetic population.""")

    tau_i = NArray(
        label=":math:`b`",
        default=numpy.array([20.0]),
        domain=Range(lo=12.0, hi=40.0, step=0.2),
        doc="""Inhibitory time constant (ms) in the first kinetic population.""")

    v0 = NArray(
        label=":math:`v_0`",
        default=numpy.array([6.0]),
        domain=Range(lo=3.12, hi=6.0, step=0.02),
        doc="""Firing threshold (PSP) for which a 50% firing rate is achieved.
        In other words, it is the value of the average membrane potential
        corresponding to the inflection point of the sigmoid [mV]; 6.0 in JansenRit1995 and DavidFriston2003""")

    e0 = NArray(
        label=":math:`e_0`",
        default=numpy.array([0.005]),
        domain=Range(lo=0.00125, hi=0.00375, step=0.00001),
        doc="""Determines the maximum firing rate of the neural population
        [ms^-1]. In the papers its value is 2 * 2.5 [s^-1]; we convert to [ms^-1]""")

    r = NArray(
        label=":math:`r`",
        default=numpy.array([0.56]),
        domain=Range(lo=0.28, hi=0.84, step=0.01),
        doc="""Steepness of the sigmoidal transformation [mV^-1].""")

    c = NArray(
        label=r":math:`c`",
        default=numpy.array([135.0]),
        domain=Range(lo=65.0, hi=1350.0, step=1.),
        doc="""Average number of synapses between populations.""")

    c_pyr2exc = NArray(
        label=r":math:`c_pyr2exc`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.5, hi=1.5, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback excitatory loop. From pyramidal cells to 
        excitatory interneurons. It multiplies c; so c_11 = c = 135 contacts""")

    c_exc2pyr = NArray(
        label=r":math:`c_exc2pyr`",
        default=numpy.array([0.8]),
        domain=Range(lo=0.4, hi=1.2, step=0.1),
        doc="""Average probability of synaptic contacts in the slow feedback excitatory loop. From excitatory 
        interneurons to pyramidal cells. It multiplies c; so c_12 = c * 0.8 = 108 contacts""")

    c_pyr2inh = NArray(
        label=r":math:`c_pyr2inh`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the feedback inhibitory loop. From pyramidal cells to 
        inhibitory interneurons. It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    c_inh2pyr = NArray(
        label=r":math:`c_inh2pyr`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the slow feedback inhibitory loop. From inhibitory cells
        to pyramidal cells. It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    p_min = NArray(
        label=":math:`p_{min}`",
        default=numpy.array([0.12]),
        domain=Range(lo=0.0, hi=0.12, step=0.01),
        doc="""Minimum input firing rate.""")

    p_max = NArray(
        label=":math:`p_{max}`",
        default=numpy.array([0.32]),
        domain=Range(lo=0.0, hi=0.32, step=0.01),
        doc="""Maximum input firing rate.""")

    p = NArray(
        label=r":math:`\mu_{max}`",
        default=numpy.array([0.22]),
        domain=Range(lo=0.0, hi=0.22, step=0.01),
        doc="""Mean input firing rate""")


    ## WILSON-COWAN parameters
    # Define traited attributes for this model, these represent possible kwargs.
    c_ee = NArray(
        label=":math:`c_{ee}`",
        default=numpy.array([12.0]),
        domain=Range(lo=11.0, hi=16.0, step=0.01),
        doc="""Excitatory to excitatory  coupling coefficient""")

    c_ei = NArray(
        label=":math:`c_{ei}`",
        default=numpy.array([4.0]),
        domain=Range(lo=2.0, hi=15.0, step=0.01),
        doc="""Inhibitory to excitatory coupling coefficient""")

    c_ie = NArray(
        label=":math:`c_{ie}`",
        default=numpy.array([13.0]),
        domain=Range(lo=2.0, hi=22.0, step=0.01),
        doc="""Excitatory to inhibitory coupling coefficient.""")

    c_ii = NArray(
        label=":math:`c_{ii}`",
        default=numpy.array([11.0]),
        domain=Range(lo=2.0, hi=15.0, step=0.01),
        doc="""Inhibitory to inhibitory coupling coefficient.""")

    tau_e_wc = NArray(
        label=r":math:`\tau_e`",
        default=numpy.array([10.0]),
        domain=Range(lo=0.0, hi=150.0, step=0.01),
        doc="""Excitatory population, membrane time-constant [ms]""")

    tau_i_wc = NArray(
        label=r":math:`\tau_i`",
        default=numpy.array([10.0]),
        domain=Range(lo=0.0, hi=150.0, step=0.01),
        doc="""Inhibitory population, membrane time-constant [ms]""")

    a_e = NArray(
        label=":math:`a_e`",
        default=numpy.array([1.2]),
        domain=Range(lo=0.0, hi=1.4, step=0.01),
        doc="""The slope parameter for the excitatory response function""")

    b_e = NArray(
        label=":math:`b_e`",
        default=numpy.array([2.8]),
        domain=Range(lo=1.4, hi=6.0, step=0.01),
        doc="""Position of the maximum slope of the excitatory sigmoid function""")

    c_e = NArray(
        label=":math:`c_e`",
        default=numpy.array([1.0]),
        domain=Range(lo=1.0, hi=20.0, step=1.0),
        doc="""The amplitude parameter for the excitatory response function""")

    theta_e = NArray(
        label=r":math:`\theta_e`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=60., step=0.01),
        doc="""Excitatory threshold""")

    a_i = NArray(
        label=":math:`a_i`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=2.0, step=0.01),
        doc="""The slope parameter for the inhibitory response function""")

    b_i = NArray(
        label=r":math:`b_i`",
        default=numpy.array([4.0]),
        domain=Range(lo=2.0, hi=6.0, step=0.01),
        doc="""Position of the maximum slope of a sigmoid function [in
           threshold units]""")

    theta_i = NArray(
        label=r":math:`\theta_i`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=60.0, step=0.01),
        doc="""Inhibitory threshold""")

    c_i = NArray(
        label=":math:`c_i`",
        default=numpy.array([1.0]),
        domain=Range(lo=1.0, hi=20.0, step=1.0),
        doc="""The amplitude parameter for the inhibitory response function""")

    r_e = NArray(
        label=":math:`r_e`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.5, hi=2.0, step=0.01),
        doc="""Excitatory refractory period""")

    r_i = NArray(
        label=":math:`r_i`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.5, hi=2.0, step=0.01),
        doc="""Inhibitory refractory period""")

    k_e = NArray(
        label=":math:`k_e`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.5, hi=2.0, step=0.01),
        doc="""Maximum value of the excitatory response function""")

    k_i = NArray(
        label=":math:`k_i`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=2.0, step=0.01),
        doc="""Maximum value of the inhibitory response function""")

    P = NArray(
        label=":math:`P`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=20.0, step=0.01),
        doc="""External stimulus to the excitatory population.
           Constant intensity.Entry point for coupling.""")

    sigma = NArray(
        label=r":math:`\sigma`",
        default=numpy.array([0.022]),
        domain=Range(lo=0.0, hi=0.05, step=0.005),
        doc="""Standard deviation of Excitatory input firing rate following a Gaussian""")

    Q = NArray(
        label=":math:`Q`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=20.0, step=0.01),
        doc="""External stimulus to the inhibitory population.
           Constant intensity.Entry point for coupling.""")

    alpha_e = NArray(
        label=r":math:`\alpha_e`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=20.0, step=0.01),
        doc="""External stimulus to the excitatory population.
           Constant intensity.Entry point for coupling.""")

    alpha_i = NArray(
        label=r":math:`\alpha_i`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=20.0, step=0.01),
        doc="""External stimulus to the inhibitory population.
           Constant intensity.Entry point for coupling.""")

    shift_sigmoid = NArray(
        dtype=numpy.bool,
        label=r":math:`shift sigmoid`",
        default=numpy.array([True]),
        doc="""In order to have resting state (E=0 and I=0) in absence of external input,
           the logistic curve are translated downward S(0)=0""")

    jrMask_wc = NArray(
        label=r":math:`\mu_{max}`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=0.1, step=0.1),
        doc="""Boolean mask for JR==True & WC==False regions""")


    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={

            # JR variables
            "vPyr": numpy.array([-1.0, 1.0]),
            "xPyr": numpy.array([-6.0, 6.0]),
            "vExc": numpy.array([-1.0, 1.0]),
            "xExc": numpy.array([-2.0, 2.0]),
            "vInh": numpy.array([-5.0, 5.0]),
            "xInh": numpy.array([-5.0, 5.0]),

            # WC variables
            "E": numpy.array([0.0, 1.0]),
            "I": numpy.array([0.0, 1.0])},

        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current
        parameters, it is used as a mechanism for bounding random inital
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("vPyr", "xPyr", "vExc", "xExc", "vInh", "xInh", "E", "I"),
        default=("vPyr", "vExc", "vInh", "E"),
        doc="""This represents the default state-variables of this Model to be
                                    monitored. It can be overridden for each Monitor if desired. The
                                    corresponding state-variable indices for this model are :math:`y0 = 0`,
                                    :math:`y1 = 1`, :math:`y2 = 2`, :math:`y3 = 3`, :math:`y4 = 4`, and
                                    :math:`y5 = 5`""")

    state_variables = ["vPyr", "xPyr", "vExc", "xExc", "vInh", "xInh", "E", "I"]
    _nvar = 8
    cvar = numpy.array([2, 4, 6, 7], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):

        vPyr = state_variables[0, :]
        xPyr = state_variables[1, :]
        vExc = state_variables[2, :]
        xExc = state_variables[3, :]
        vInh = state_variables[4, :]
        xInh = state_variables[5, :]

        E = state_variables[6, :]
        I = state_variables[7, :]

        lrc_2jr = coupling[0, :] * self.jrMask_wc[numpy.newaxis].T
        lrc_2wc = coupling[0, :] * (1-self.jrMask_wc)[numpy.newaxis].T

        ## JANSEN-RIT
        src = local_coupling * (vExc - vInh)

        S_pyr = (self.e0) / (1 + numpy.exp(self.r * (self.v0 - (vExc - vInh))))
        S_exc = (self.c * self.c_exc2pyr * self.e0) / (1 + numpy.exp(self.r * (self.v0 - self.c * self.c_pyr2exc * vPyr)))
        S_inh = (self.c * self.c_inh2pyr * self.e0) / (1 + numpy.exp(self.r * (self.v0 - self.c * self.c_pyr2inh * vPyr)))

        dvPyr = xPyr
        dxPyr = self.He / self.tau_e * S_pyr - (2 * xPyr) / self.tau_e - (vPyr / self.tau_e**2)
        dvExc = xExc
        dxExc = self.He / self.tau_e * (S_exc + src + lrc_2jr + self.p) - (2 * xExc) / self.tau_e - (vExc / self.tau_e**2)
        dvInh = xInh
        dxInh = self.Hi / self.tau_i * S_inh - (2 * xInh) / self.tau_i - (vInh / self.tau_i**2)

        ## WILSON-COWAN
        # short-range (local) coupling
        src_e = local_coupling * E
        src_i = local_coupling * I

        ## Extrinsic (sensory & random) input
        input = numpy.random.normal(self.P, self.sigma, (len(coupling[0]), 1))

        x_e = self.alpha_e * (self.c_ee * E - self.c_ei * I + input - self.theta_e + lrc_2wc + src_e + src_i)
        x_i = self.alpha_i * (self.c_ie * E - self.c_ii * I + self.Q - self.theta_i + src_e + src_i)

        s_e = self.c_e / (1.0 + numpy.exp(-self.a_e * (x_e - self.b_e)))
        s_i = self.c_i / (1.0 + numpy.exp(-self.a_i * (x_i - self.b_i)))

        dE = (-E + (self.k_e - self.r_e * E) * s_e) / self.tau_e_wc
        dI = (-I + (self.k_i - self.r_i * I) * s_i) / self.tau_i_wc

        derivative = numpy.array([dvPyr, dxPyr, dvExc, dxExc, dvInh, dxInh, dE, dI])

        return derivative
