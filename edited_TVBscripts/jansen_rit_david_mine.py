# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
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
#
#

"""
A contributed model: The Jansen and Rit model as presented in (David et al., 2003) [JansenRitDavid2003];
and an extension to build a thalamocortical network combining the hierarchical modeling approach in
David et al. (2005) and the differential thalamo-cortical and cortico-cortical connectivity role
as proposed in Jones (2009).

@01/08/2022 Adding Wendling (2002) model for epilepsy.

.. moduleauthor:: Jesús Cabrera-Álvarez (jescab01@ucm.es)

"""

import numpy

from tvb.simulator.common import get_logger
from tvb.basic.neotraits.api import NArray, Range, Final, List
import tvb.simulator.models as models

LOG = get_logger(__name__)



class JansenRit1995(models.Model):
    """
    This is a version of TVB Jansen-Rit model; in which I just change variable names
    to get an easier to read model; as baseline for further work on it:
        - adding specific noise to thalamus
        - building JansenRitDavid model above.
    """

    # Define traited attributes for this model, these represent possible kwargs.
    He = NArray(
        label=":math:`He`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain.""")

    Hi = NArray(
        label=":math:`B`",
        default=numpy.array([22.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain.""")

    tau_e = NArray(
        label=":math:`a`",
        default=numpy.array([10.0]),
        domain=Range(lo=6.3, hi=20.0, step=0.1),
        doc="""Excitatory time constant (ms)""")

    tau_i = NArray(
        label=":math:`b`",
        default=numpy.array([20.0]),
        domain=Range(lo=12.0, hi=40.0, step=0.2),
        doc="""Inhibitory time constant (ms)""")

    v0 = NArray(
        label=":math:`v_0`",
        default=numpy.array([5.52]),
        domain=Range(lo=3.12, hi=6.0, step=0.02),
        doc="""Firing threshold (PSP) for which a 50% firing rate is achieved.
        In other words, it is the value of the average membrane potential
        corresponding to the inflection point of the sigmoid [mV]; 
        6.0 in JansenRit1995 and DavidFriston2003""")

    e0 = NArray(
        label=":math:`e_0`",
        default=numpy.array([0.005]),
        domain=Range(lo=0.00125, hi=0.00375, step=0.00001),
        doc="""Determines the maximum firing rate of the neural population
        [ms^-1]. In the papers its value is 2.5 [s^-1]; we convert to [ms^-1]""")

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
        label=r":math:`c_11`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.5, hi=1.5, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback excitatory loop.
        It multiplies c; so c_11 = c = 135 contacts""")

    c_exc2pyr = NArray(
        label=r":math:`c_12`",
        default=numpy.array([0.8]),
        domain=Range(lo=0.4, hi=1.2, step=0.1),
        doc="""Average probability of synaptic contacts in the slow feedback excitatory loop. 
        It multiplies c; so c_12 = c * 0.8 = 108 contacts""")

    c_pyr2inh = NArray(
        label=r":math:`c_21`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the feedback inhibitory loop.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    c_inh2pyr = NArray(
        label=r":math:`c_22`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the slow feedback inhibitory loop.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    p = NArray(
        label=r":math:`\p_{max}`",
        default=numpy.array([0.22]),
        domain=Range(lo=0.0, hi=0.22, step=0.01),
        doc="""Mean input firing rate""")

    sigma = NArray(
        label=r":math:`\sigma`",
        default=numpy.array([0.022]),
        domain=Range(lo=0.0, hi=0.05, step=0.005),
        doc="""Standard deviation of input firing rate following a Gaussian""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"vPyr": numpy.array([-1.0, 1.0]),
                 "vExc": numpy.array([-500.0, 500.0]),
                 "vInh": numpy.array([-50.0, 50.0]),
                 "xPyr": numpy.array([-6.0, 6.0]),
                 "xExc": numpy.array([-20.0, 20.0]),
                 "xInh": numpy.array([-500.0, 500.0])},
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current 
        parameters, it is used as a mechanism for bounding random inital 
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("vPyr", "vExc", "vInh", "xPyr", "xExc", "xInh"),
        default=("vPyr", "vExc", "vInh", "xPyr"),
        doc="""This represents the default state-variables of this Model to be
                                    monitored. It can be overridden for each Monitor if desired. The 
                                    corresponding state-variable indices for this model are :math:`y0 = 0`,
                                    :math:`y1 = 1`, :math:`y2 = 2`, :math:`y3 = 3`, :math:`y4 = 4`, and
                                    :math:`y5 = 5`""")

    state_variables = ["vPyr", "vExc", "vInh", "xPyr", "xExc", "xInh"]
    _nvar = 6
    cvar = numpy.array([1, 2], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The dynamic equations were taken from:
        TODO: add equations and finish the model ...
        """

        vPyr = state_variables[0, :]
        vExc = state_variables[1, :]
        vInh = state_variables[2, :]
        xPyr = state_variables[3, :]
        xExc = state_variables[4, :]
        xInh = state_variables[5, :]

        # NOTE: This is assumed to be \sum_j u_kj * S[y_{1_j} - y_{2_j}]
        lrc = coupling[0, :]
        src = local_coupling * (vExc - vInh)

        ## Intrinsic input
        input = numpy.random.normal(self.p, self.sigma, (len(coupling[0]), 1))

        S_pyr = self.e0 / (1 + numpy.exp(self.r * (self.v0 - (vExc - vInh))))
        S_exc = self.e0 / (1 + numpy.exp(self.r * (self.v0 - self.c_pyr2exc * self.c * vPyr)))
        S_inh = self.e0 / (1 + numpy.exp(self.r * (self.v0 - self.c_pyr2inh * self.c * vPyr)))

        ## NOTE, for local couplings:
        # vExc, xExc excitatory interneurons - gate for intrinsic (p) and external (lrc, src) inputs.
        # vInh, xInh inhibitory interneurons
        # vPyr, xPyr pyramidal neurons

        dvPyr = xPyr
        dvExc = xExc
        dvInh = xInh
        dxPyr = self.He / self.tau_e * S_pyr - (2 * xPyr) / self.tau_e - vPyr / self.tau_e ** 2
        dxExc = self.He / self.tau_e * (input + self.c_exc2pyr * self.c * S_exc + lrc + src) - (2 * xExc) / self.tau_e - vExc / self.tau_e ** 2
        dxInh = self.Hi / self.tau_i * (self.c_inh2pyr * self.c * S_inh) - (2 * xInh) / self.tau_i - vInh / self.tau_i ** 2

        derivative = numpy.array([dvPyr, dvExc, dvInh, dxPyr, dxExc, dxInh])

        return derivative

class JansenRit1995_sumNtt5(models.Model):
    """
    This is a version of TVB Jansen-Rit model;
     with a neurotransmission system implemented with its own dynamics and parameters.
     "sum" refers to the way in which neuromodulation affects neural activity. In this case,
     it is summed to the general inputs following Kringelbach 2020
    """

    # JR parameters
    He = NArray(
        label=":math:`He`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain.""")

    Hi = NArray(
        label=":math:`B`",
        default=numpy.array([22.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain.""")

    tau_e = NArray(
        label=":math:`a`",
        default=numpy.array([10.0]),
        domain=Range(lo=6.3, hi=20.0, step=0.1),
        doc="""Excitatory time constant (ms)""")

    tau_i = NArray(
        label=":math:`b`",
        default=numpy.array([20.0]),
        domain=Range(lo=12.0, hi=40.0, step=0.2),
        doc="""Inhibitory time constant (ms)""")

    v0 = NArray(
        label=":math:`v_0`",
        default=numpy.array([5.52]),
        domain=Range(lo=3.12, hi=6.0, step=0.02),
        doc="""Firing threshold (PSP) for which a 50% firing rate is achieved.
        In other words, it is the value of the average membrane potential
        corresponding to the inflection point of the sigmoid [mV]; 
        6.0 in JansenRit1995 and DavidFriston2003""")

    e0 = NArray(
        label=":math:`e_0`",
        default=numpy.array([0.005]),
        domain=Range(lo=0.00125, hi=0.00375, step=0.00001),
        doc="""Determines the maximum firing rate of the neural population
        [ms^-1]. In the papers its value is 2.5 [s^-1]; we convert to [ms^-1]""")

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
        label=r":math:`c_11`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.5, hi=1.5, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback excitatory loop.
        It multiplies c; so c_11 = c = 135 contacts""")

    c_exc2pyr = NArray(
        label=r":math:`c_12`",
        default=numpy.array([0.8]),
        domain=Range(lo=0.4, hi=1.2, step=0.1),
        doc="""Average probability of synaptic contacts in the slow feedback excitatory loop. 
        It multiplies c; so c_12 = c * 0.8 = 108 contacts""")

    c_pyr2inh = NArray(
        label=r":math:`c_21`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the feedback inhibitory loop.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    c_inh2pyr = NArray(
        label=r":math:`c_22`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the slow feedback inhibitory loop.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    p = NArray(
        label=r":math:`\p_{max}`",
        default=numpy.array([0.22]),
        domain=Range(lo=0.0, hi=0.22, step=0.01),
        doc="""Mean input firing rate""")

    sigma = NArray(
        label=r":math:`\sigma`",
        default=numpy.array([0.022]),
        domain=Range(lo=0.0, hi=0.05, step=0.005),
        doc="""Standard deviation of input firing rate following a Gaussian""")


    ## Neurotransmission parameters
    eta = NArray(
        label=r":math:`\eta`",
        default=numpy.array([0.5,]),
        domain=Range(lo=0, hi=1000, step=1),
        doc="Decay rate for simple neurotransmission reuptake implementation",)

    tau_m = NArray(
        label=":math:`b`",
        default=numpy.array([120]),
        domain=Range(lo=12.0, hi=240.0, step=0.2),
        doc="""Time constant (s) for the effect of the neurotransmitter """)

    receptormaps = NArray(
        label=r":math:`ntt_ids`",
        default=numpy.array([0.5,]),
        domain=Range(lo=0, hi=1000, step=1),
        doc="Mask for source neurotransmission ROIs",)


    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"vPyr": numpy.array([-1.0, 1.0]),
                 "vExc": numpy.array([-500.0, 500.0]),
                 "vInh": numpy.array([-50.0, 50.0]),
                 "xPyr": numpy.array([-6.0, 6.0]),
                 "xExc": numpy.array([-20.0, 20.0]),
                 "xInh": numpy.array([-500.0, 500.0]),

                 "S_5HT":numpy.array([0, 0.5]),
                 "S_NE": numpy.array([0, 0.5]),
                 "S_D": numpy.array([0, 0]),
                 "S_ACh": numpy.array([0, 0]),
                 "S_Glu": numpy.array([0, 0]),

                 "M_5HT": numpy.array([0, 0]),
                 "M_NE": numpy.array([0, 0]),
                 "M_D": numpy.array([0, 0]),
                 "M_ACh": numpy.array([0, 0]),
                 "M_Glu": numpy.array([0, 0])},

        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current 
        parameters, it is used as a mechanism for bounding random inital 
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("vPyr", "vExc", "vInh", "xPyr", "xExc", "xInh",
                 "S_5HT", "S_NE", "S_D", "S_ACh", "S_Glu",
                 "M_5HT", "M_NE", "M_D", "M_ACh", "M_Glu"),

        default=("vPyr", "vExc", "vInh", "xPyr", "xExc", "xInh", "S_5HT", "M_5HT"),
        doc="""This represents the default state-variables of this Model to be
                                    monitored. It can be overridden for each Monitor if desired. The 
                                    corresponding state-variable indices for this model are :math:`y0 = 0`,
                                    :math:`y1 = 1`, :math:`y2 = 2`, :math:`y3 = 3`, :math:`y4 = 4`, and
                                    :math:`y5 = 5`""")

    state_variables = ["vPyr", "vExc", "vInh", "xPyr", "xExc", "xInh",
                       "S_5HT", "S_NE", "S_D", "S_ACh", "S_Glu",
                       "M_5HT", "M_NE", "M_D", "M_ACh", "M_Glu"]
    _nvar = 16
    cvar = numpy.array([1, 2, 3, 4, 5, 6], dtype=numpy.int32)  # To be defined from outside after model definition: 1 (normal coupling) + n Neurotransmitters

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The dynamic equations were taken from:
        TODO: add equations and finish the model ...
        """

        vPyr = state_variables[0, :]
        vExc = state_variables[1, :]
        vInh = state_variables[2, :]
        xPyr = state_variables[3, :]
        xExc = state_variables[4, :]
        xInh = state_variables[5, :]

        S_5HT = state_variables[6, :]  # Neurotransmitter concentration
        S_NE = state_variables[7, :]
        S_D = state_variables[8, :]
        S_ACh = state_variables[9, :]
        S_Glu = state_variables[10, :]

        M_5HT = state_variables[11, :]  # Modulation: generated Firing rate
        M_NE = state_variables[12, :]
        M_D = state_variables[13, :]
        M_ACh = state_variables[14, :]
        M_Glu = state_variables[15, :]


        ## Intrinsic input
        input = numpy.random.normal(self.p, self.sigma, (len(coupling[0]), 1))
        ## Neuromodulation input
        mod = M_5HT + M_NE + M_D + M_ACh + M_Glu
        # NOTE: This is assumed to be \sum_j u_kj * S[y_{1_j} - y_{2_j}]
        src = local_coupling * (vExc - vInh)
        lrc = coupling[0, :]  # Global coupling: sum of firing rates coming into a population

        S_pyr = self.e0 / (1 + numpy.exp(self.r * (self.v0 - (vExc - vInh))))
        S_exc = self.e0 / (1 + numpy.exp(self.r * (self.v0 - self.c_pyr2exc * self.c * vPyr)))
        S_inh = self.e0 / (1 + numpy.exp(self.r * (self.v0 - self.c_pyr2inh * self.c * vPyr)))


        ## NOTE, for local couplings:
        # vExc, xExc excitatory interneurons - gate for intrinsic (p) and external (lrc, src) inputs.
        # vInh, xInh inhibitory interneurons; vPyr, xPyr pyramidal neurons
        dvPyr = xPyr
        dvExc = xExc
        dvInh = xInh
        dxPyr = self.He / self.tau_e * (S_pyr + mod) - (2 * xPyr) / self.tau_e - vPyr / self.tau_e ** 2
        dxExc = self.He / self.tau_e * (self.c_exc2pyr * self.c * S_exc + input + lrc + src + mod) - (2 * xExc) / self.tau_e - vExc / self.tau_e ** 2
        dxInh = self.Hi / self.tau_i * (self.c_inh2pyr * self.c * S_inh + mod) - (2 * xInh) / self.tau_i - vInh / self.tau_i ** 2

        lrm = coupling[1:, :]  # Neurotransmission coupling (ntt, roi, 1): sum of firing rates from neurotransmission sources (e.g. raphe)
        ## Ntt concentrations: using the simplest decay used in Joshy (2016) for orexin (eq. 2.6)
        dS_5HT = lrm[0, :] - self.eta[0] * S_5HT    if len(self.eta) > 1 else lrm[0, :] - self.eta * S_5HT
        dS_NE  = lrm[1, :] - self.eta[1] * S_NE   if len(self.eta) > 1 else lrm[1, :] - self.eta * S_NE
        dS_D = lrm[2, :] - self.eta[2] * S_D if len(self.eta) > 1 else lrm[2, :] - self.eta * S_D
        dS_ACh = lrm[3, :] - self.eta[3] * S_ACh if len(self.eta) > 1 else lrm[3, :] - self.eta * S_ACh
        dS_Glu = lrm[4, :] - self.eta[4] * S_Glu if len(self.eta) > 1 else lrm[4, :] - self.eta * S_Glu

        ## Modulation - Ntt impact on firing rate: using JR sigmoidal. Applying receptor maps (ntt, rois) following Kringelbach 20220 (eq. 16).
        dM_5HT = (-M_5HT + (self.e0 / (1 + numpy.exp(-self.r * (numpy.log10(numpy.array([self.receptormaps[:, 0]]).T * S_5HT) + 1))))) / self.tau_m
        dM_NE = (-M_NE + (self.e0 / (1 + numpy.exp(-self.r * (numpy.log10(numpy.array([self.receptormaps[:, 1]]).T * S_NE) + 1))))) / self.tau_m
        dM_D = (-M_D + (self.e0 / (1 + numpy.exp(-self.r * (numpy.log10(numpy.array([self.receptormaps[:, 2]]).T * S_D) + 1))))) / self.tau_m
        dM_ACh = (-M_ACh + (self.e0 / (1 + numpy.exp(-self.r * (numpy.log10(numpy.array([self.receptormaps[:, 3]]).T * S_ACh) + 1))))) / self.tau_m
        dM_Glu = (-M_Glu + (self.e0 / (1 + numpy.exp(-self.r * (numpy.log10(numpy.array([self.receptormaps[:, 4]]).T * S_Glu) + 1))))) / self.tau_m

        derivative = numpy.array([dvPyr, dvExc, dvInh, dxPyr, dxExc, dxInh, dS_5HT, dS_NE, dS_D, dS_ACh, dS_Glu, dM_5HT, dM_NE, dM_D, dM_ACh, dM_Glu])

        return derivative

class JansenRit1995_gainNtt5(models.Model):
    """
    This is a version of TVB Jansen-Rit model;
     with a neurotransmission system implemented with its own dynamics and parameters.
     "gain" refers to the way in which the neuromodulation affects neuronal activity. In this case, it
     applies a gain into firing rate computation following Deco (2018).
    """

    # JR parameters
    He = NArray(
        label=":math:`He`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain.""")

    Hi = NArray(
        label=":math:`B`",
        default=numpy.array([22.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain.""")

    tau_e = NArray(
        label=":math:`a`",
        default=numpy.array([10.0]),
        domain=Range(lo=6.3, hi=20.0, step=0.1),
        doc="""Excitatory time constant (ms)""")

    tau_i = NArray(
        label=":math:`b`",
        default=numpy.array([20.0]),
        domain=Range(lo=12.0, hi=40.0, step=0.2),
        doc="""Inhibitory time constant (ms)""")

    v0 = NArray(
        label=":math:`v_0`",
        default=numpy.array([5.52]),
        domain=Range(lo=3.12, hi=6.0, step=0.02),
        doc="""Firing threshold (PSP) for which a 50% firing rate is achieved.
        In other words, it is the value of the average membrane potential
        corresponding to the inflection point of the sigmoid [mV]; 
        6.0 in JansenRit1995 and DavidFriston2003""")

    e0 = NArray(
        label=":math:`e_0`",
        default=numpy.array([0.005]),
        domain=Range(lo=0.00125, hi=0.00375, step=0.00001),
        doc="""Determines the maximum firing rate of the neural population
        [ms^-1]. In the papers its value is 2.5 [s^-1]; we convert to [ms^-1]""")

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
        label=r":math:`c_11`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.5, hi=1.5, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback excitatory loop.
        It multiplies c; so c_11 = c = 135 contacts""")

    c_exc2pyr = NArray(
        label=r":math:`c_12`",
        default=numpy.array([0.8]),
        domain=Range(lo=0.4, hi=1.2, step=0.1),
        doc="""Average probability of synaptic contacts in the slow feedback excitatory loop. 
        It multiplies c; so c_12 = c * 0.8 = 108 contacts""")

    c_pyr2inh = NArray(
        label=r":math:`c_21`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the feedback inhibitory loop.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    c_inh2pyr = NArray(
        label=r":math:`c_22`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the slow feedback inhibitory loop.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    p = NArray(
        label=r":math:`\p_{max}`",
        default=numpy.array([0.22]),
        domain=Range(lo=0.0, hi=0.22, step=0.01),
        doc="""Mean input firing rate""")

    sigma = NArray(
        label=r":math:`\sigma`",
        default=numpy.array([0.022]),
        domain=Range(lo=0.0, hi=0.05, step=0.005),
        doc="""Standard deviation of input firing rate following a Gaussian""")


    ## Neurotransmission parameters
    eta = NArray(
        label=r":math:`\eta`",
        default=numpy.array([0.5,]),
        domain=Range(lo=0, hi=1000, step=1),
        doc="Decay rate for simple neurotransmission reuptake implementation",)

    tau_m = NArray(
        label=":math:`b`",
        default=numpy.array([120]),
        domain=Range(lo=12.0, hi=240.0, step=0.2),
        doc="""Time constant (ms) for the effect of the neurotransmitter """)

    receptormaps = NArray(
        label=r":math:`ntt_ids`",
        default=numpy.array([0.5,]),
        domain=Range(lo=0, hi=1000, step=1),
        doc="Mask for source neurotransmission ROIs",)


    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"vPyr": numpy.array([-1.0, 1.0]),
                 "vExc": numpy.array([-500.0, 500.0]),
                 "vInh": numpy.array([-50.0, 50.0]),
                 "xPyr": numpy.array([-6.0, 6.0]),
                 "xExc": numpy.array([-20.0, 20.0]),
                 "xInh": numpy.array([-500.0, 500.0]),

                 "S_5HT":numpy.array([0, 0.5]),
                 "S_NE": numpy.array([0, 0.5]),
                 "S_D": numpy.array([0, 0]),
                 "S_ACh": numpy.array([0, 0]),
                 "S_Glu": numpy.array([0, 0]),

                 "M_5HT": numpy.array([0, 0]),
                 "M_NE": numpy.array([0, 0]),
                 "M_D": numpy.array([0, 0]),
                 "M_ACh": numpy.array([0, 0]),
                 "M_Glu": numpy.array([0, 0])},

        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current 
        parameters, it is used as a mechanism for bounding random inital 
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("vPyr", "vExc", "vInh", "xPyr", "xExc", "xInh",
                 "S_5HT", "S_NE", "S_D", "S_ACh", "S_Glu",
                 "M_5HT", "M_NE", "M_D", "M_ACh", "M_Glu"),

        default=("vPyr", "vExc", "vInh", "xPyr", "xExc", "xInh", "S_5HT", "M_5HT"),
        doc="""This represents the default state-variables of this Model to be
                                    monitored. It can be overridden for each Monitor if desired. The 
                                    corresponding state-variable indices for this model are :math:`y0 = 0`,
                                    :math:`y1 = 1`, :math:`y2 = 2`, :math:`y3 = 3`, :math:`y4 = 4`, and
                                    :math:`y5 = 5`""")

    state_variables = ["vPyr", "vExc", "vInh", "xPyr", "xExc", "xInh",
                       "S_5HT", "S_NE", "S_D", "S_ACh", "S_Glu",
                       "M_5HT", "M_NE", "M_D", "M_ACh", "M_Glu"]
    _nvar = 16
    cvar = numpy.array([1, 2, 3, 4, 5, 6], dtype=numpy.int32)  # To be defined from outside after model definition: 1 (normal coupling) + n Neurotransmitters

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The dynamic equations were taken from:
        TODO: add equations and finish the model ...
        """

        vPyr = state_variables[0, :]
        vExc = state_variables[1, :]
        vInh = state_variables[2, :]
        xPyr = state_variables[3, :]
        xExc = state_variables[4, :]
        xInh = state_variables[5, :]

        S_5HT = state_variables[6, :]  # Neurotransmitter concentration
        S_NE = state_variables[7, :]
        S_D = state_variables[8, :]
        S_ACh = state_variables[9, :]
        S_Glu = state_variables[10, :]

        M_5HT = state_variables[11, :]  # Modulation: generated Firing rate
        M_NE = state_variables[12, :]
        M_D = state_variables[13, :]
        M_ACh = state_variables[14, :]
        M_Glu = state_variables[15, :]


        ## Intrinsic input
        input = numpy.random.normal(self.p, self.sigma, (len(coupling[0]), 1))
        ## Neuromodulation input
        mod = M_5HT + M_NE + M_D + M_ACh + M_Glu
        # NOTE: This is assumed to be \sum_j u_kj * S[y_{1_j} - y_{2_j}]
        src = local_coupling * (vExc - vInh)
        lrc = coupling[0, :]  # Global coupling: sum of firing rates coming into a population

        S_pyr = self.e0 / (1 + numpy.exp(self.r * mod * (self.v0 - (vExc - vInh))))
        S_exc = self.e0 / (1 + numpy.exp(self.r * mod * (self.v0 - self.c_pyr2exc * self.c * vPyr)))
        S_inh = self.e0 / (1 + numpy.exp(self.r * (self.v0 - self.c_pyr2inh * self.c * vPyr)))


        ## NOTE, for local couplings:
        # vExc, xExc excitatory interneurons - gate for intrinsic (p) and external (lrc, src) inputs.
        # vInh, xInh inhibitory interneurons; vPyr, xPyr pyramidal neurons
        dvPyr = xPyr
        dvExc = xExc
        dvInh = xInh
        dxPyr = self.He / self.tau_e * (S_pyr) - (2 * xPyr) / self.tau_e - vPyr / self.tau_e ** 2
        dxExc = self.He / self.tau_e * (self.c_exc2pyr * self.c * S_exc + (input + lrc + src) * mod) - (2 * xExc) / self.tau_e - vExc / self.tau_e ** 2
        dxInh = self.Hi / self.tau_i * (self.c_inh2pyr * self.c * S_inh) - (2 * xInh) / self.tau_i - vInh / self.tau_i ** 2

        lrm = coupling[1:, :]  # Neurotransmission coupling (ntt, roi, 1): sum of firing rates from neurotransmission sources (e.g. raphe)
        ## Ntt concentrations: using the simplest decay used in Joshy (2016) for orexin (eq. 2.6)
        dS_5HT = lrm[0, :] - self.eta[0] * S_5HT    if len(self.eta) > 1 else lrm[0, :] - self.eta * S_5HT
        dS_NE  = lrm[1, :] - self.eta[1] * S_NE   if len(self.eta) > 1 else lrm[1, :] - self.eta * S_NE
        dS_D = lrm[2, :] - self.eta[2] * S_D if len(self.eta) > 1 else lrm[2, :] - self.eta * S_D
        dS_ACh = lrm[3, :] - self.eta[3] * S_ACh if len(self.eta) > 1 else lrm[3, :] - self.eta * S_ACh
        dS_Glu = lrm[4, :] - self.eta[4] * S_Glu if len(self.eta) > 1 else lrm[4, :] - self.eta * S_Glu

        ## Modulation - Ntt impact on firing rate: using JR sigmoidal. Applying receptor maps (ntt, rois)

        dM_5HT = (-M_5HT + (numpy.array([self.receptormaps[:, 0]]).T *S_5HT ) ) / self.tau_m

        # dM_5HT = (-M_5HT + (self.e0 / (1 + numpy.exp(-self.r * (numpy.log10(numpy.array([self.receptormaps[:, 0]]).T * S_5HT) + 1))))) / self.tau_m
        # dM_NE = (-M_NE + (self.e0 / (1 + numpy.exp(-self.r * (numpy.log10(numpy.array([self.receptormaps[:, 1]]).T * S_NE) + 1))))) / self.tau_m
        # dM_D = (-M_D + (self.e0 / (1 + numpy.exp(-self.r * (numpy.log10(numpy.array([self.receptormaps[:, 2]]).T * S_D) + 1))))) / self.tau_m
        # dM_ACh = (-M_ACh + (self.e0 / (1 + numpy.exp(-self.r * (numpy.log10(numpy.array([self.receptormaps[:, 3]]).T * S_ACh) + 1))))) / self.tau_m
        # dM_Glu = (-M_Glu + (self.e0 / (1 + numpy.exp(-self.r * (numpy.log10(numpy.array([self.receptormaps[:, 4]]).T * S_Glu) + 1))))) / self.tau_m

        derivative = numpy.array([dvPyr, dvExc, dvInh, dxPyr, dxExc, dxInh, dS_5HT, dS_NE, dS_D, dS_ACh, dS_Glu, dM_5HT, dM_NE, dM_D, dM_ACh, dM_Glu])

        return derivative



class JansenRit1995_hierarchical(models.Model):
    """
    This is a version of TVB Jansen-Rit model in which I implement the hierarchical connectivity
    proposed by David (2005).
    """

    # Define traited attributes for this model, these represent possible kwargs.
    He = NArray(
        label=":math:`He`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain.""")

    Hi = NArray(
        label=":math:`B`",
        default=numpy.array([22.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain.""")

    tau_e = NArray(
        label=":math:`a`",
        default=numpy.array([10.0]),
        domain=Range(lo=6.3, hi=20.0, step=0.1),
        doc="""Excitatory time constant (ms)""")

    tau_i = NArray(
        label=":math:`b`",
        default=numpy.array([20.0]),
        domain=Range(lo=12.0, hi=40.0, step=0.2),
        doc="""Inhibitory time constant (ms)""")

    v0 = NArray(
        label=":math:`v_0`",
        default=numpy.array([5.52]),
        domain=Range(lo=3.12, hi=6.0, step=0.02),
        doc="""Firing threshold (PSP) for which a 50% firing rate is achieved.
        In other words, it is the value of the average membrane potential
        corresponding to the inflection point of the sigmoid [mV]; 
        6.0 in JansenRit1995 and DavidFriston2003""")

    e0 = NArray(
        label=":math:`e_0`",
        default=numpy.array([0.005]),
        domain=Range(lo=0.00125, hi=0.00375, step=0.00001),
        doc="""Determines the maximum firing rate of the neural population
        [ms^-1]. In the papers its value is 2.5 [s^-1]; we convert to [ms^-1]""")

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
        label=r":math:`c_11`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.5, hi=1.5, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback excitatory loop.
        It multiplies c; so c_11 = c = 135 contacts""")

    c_exc2pyr = NArray(
        label=r":math:`c_12`",
        default=numpy.array([0.8]),
        domain=Range(lo=0.4, hi=1.2, step=0.1),
        doc="""Average probability of synaptic contacts in the slow feedback excitatory loop. 
        It multiplies c; so c_12 = c * 0.8 = 108 contacts""")

    c_pyr2inh = NArray(
        label=r":math:`c_21`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the feedback inhibitory loop.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    c_inh2pyr = NArray(
        label=r":math:`c_22`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the slow feedback inhibitory loop.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    p = NArray(
        label=r":math:`\p_{max}`",
        default=numpy.array([0.22]),
        domain=Range(lo=0.0, hi=0.22, step=0.01),
        doc="""Mean input firing rate""")

    sigma = NArray(
        label=r":math:`\sigma`",
        default=numpy.array([0.022]),
        domain=Range(lo=0.0, hi=0.05, step=0.005),
        doc="""Standard deviation of input firing rate following a Gaussian""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"vPyr": numpy.array([-1.0, 1.0]),
                 "vExc": numpy.array([-500.0, 500.0]),
                 "vInh": numpy.array([-50.0, 50.0]),
                 "xPyr": numpy.array([-6.0, 6.0]),
                 "xExc": numpy.array([-20.0, 20.0]),
                 "xInh": numpy.array([-500.0, 500.0])},
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current 
        parameters, it is used as a mechanism for bounding random inital 
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("vPyr", "vExc", "vInh", "xPyr", "xExc", "xInh"),
        default=("vPyr", "vExc", "vInh", "xPyr"),
        doc="""This represents the default state-variables of this Model to be
                                    monitored. It can be overridden for each Monitor if desired. The 
                                    corresponding state-variable indices for this model are :math:`y0 = 0`,
                                    :math:`y1 = 1`, :math:`y2 = 2`, :math:`y3 = 3`, :math:`y4 = 4`, and
                                    :math:`y5 = 5`""")

    state_variables = ["vPyr", "vExc", "vInh", "xPyr", "xExc", "xInh"]
    _nvar = 6
    cvar = numpy.array([1, 2, 3], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The dynamic equations were taken from:
        TODO: add equations and finish the model ...
        """

        vPyr = state_variables[0, :]
        vExc = state_variables[1, :]
        vInh = state_variables[2, :]
        xPyr = state_variables[3, :]
        xExc = state_variables[4, :]
        xInh = state_variables[5, :]

        ## Long-range coupling with hierarchical connections
        # delayed activity x1 - x2
        (lrcFF, lrcFB, lrcL) = coupling

        # Short-range coupling for surface simulations
        #  TODO how to model neighbour connections ¿lateral? ¿FF?
        # srcFF = local_coupling  * (vExc - vInh) * self.aF
        # srcFB = local_coupling  * (vExc - vInh) * self.aB
        srcL = local_coupling * (vExc - vInh)   # * self.aL


        ## Intrinsic input
        input = numpy.random.normal(self.p, self.sigma, (len(coupling[0]), 1))

        S_pyr = self.e0 / (1 + numpy.exp(self.r * (self.v0 - (vExc - vInh))))
        S_exc = self.e0 / (1 + numpy.exp(self.r * (self.v0 - self.c_pyr2exc * self.c * vPyr)))
        S_inh = self.e0 / (1 + numpy.exp(self.r * (self.v0 - self.c_pyr2inh * self.c * vPyr)))

        ## NOTE, for local couplings:
        # vExc, xExc excitatory interneurons - gate for intrinsic (p) and external (lrc, src) inputs.
        # vInh, xInh inhibitory interneurons
        # vPyr, xPyr pyramidal neurons

        dvPyr = xPyr
        dvExc = xExc
        dvInh = xInh

        dxPyr = self.He / self.tau_e * (lrcFB + lrcL + srcL + S_pyr) - (2 * xPyr) / self.tau_e - vPyr / self.tau_e ** 2
        dxExc = self.He / self.tau_e * (input + lrcFF + lrcL + srcL + self.c_exc2pyr * self.c * S_exc) - (2 * xExc) / self.tau_e - vExc / self.tau_e ** 2
        dxInh = self.Hi / self.tau_i * (lrcFB + lrcL + srcL + self.c_inh2pyr * self.c * S_inh) - (2 * xInh) / self.tau_i - vInh / self.tau_i ** 2

        derivative = numpy.array([dvPyr, dvExc, dvInh, dxPyr, dxExc, dxInh])

        return derivative




class Wendling2002(models.Model):
    """
    Wendling 2002 . Epileptic fast activity can be explained by a model of impaired GABAergic dendritic inhibition.

    """

    # Define traited attributes for this model, these represent possible kwargs.
    He = NArray(
        label=":math:`He`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain.""")

    Hi = NArray(
        label=":math:`B`",
        default=numpy.array([22.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain.""")

    Hif = NArray(
        label=":math:`G`",
        default=numpy.array([10.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""fast inhibitory Maximum amplitude of IPSP [mV]. Also called average synaptic gain.""")

    tau_e = NArray(
        label=":math:`a`",
        default=numpy.array([10.0]),
        domain=Range(lo=6.3, hi=20.0, step=0.1),
        doc="""Excitatory time constant (ms)""")

    tau_i = NArray(
        label=":math:`b`",
        default=numpy.array([20.0]),
        domain=Range(lo=12.0, hi=40.0, step=0.2),
        doc="""Inhibitory time constant (ms)""")

    tau_if = NArray(
        label=":math:`g`",
        default=numpy.array([2.0]),
        domain=Range(lo=12.0, hi=40.0, step=0.2),
        doc="""fast Inhibitory time constant (ms)""")

    v0 = NArray(
        label=":math:`v_0`",
        default=numpy.array([5.52]),
        domain=Range(lo=3.12, hi=6.0, step=0.02),
        doc="""Firing threshold (PSP) for which a 50% firing rate is achieved.
        In other words, it is the value of the average membrane potential
        corresponding to the inflection point of the sigmoid [mV]; 
        6.0 in JansenRit1995 and DavidFriston2003""")

    e0 = NArray(
        label=":math:`e_0`",
        default=numpy.array([0.005]),
        domain=Range(lo=0.00125, hi=0.00375, step=0.00001),
        doc="""Determines the maximum firing rate of the neural population
        [ms^-1]. In the papers its value is 2.5 [s^-1]; we convert to [ms^-1]""")

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
        label=r":math:`c_11`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.5, hi=1.5, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback excitatory loop.
        It multiplies c; so c_11 = c = 135 contacts""")

    c_exc2pyr = NArray(
        label=r":math:`c_12`",
        default=numpy.array([0.8]),
        domain=Range(lo=0.4, hi=1.2, step=0.1),
        doc="""Average probability of synaptic contacts in the slow feedback excitatory loop. 
        It multiplies c; so c_12 = c * 0.8 = 108 contacts""")

    c_pyr2inh = NArray(
        label=r":math:`c_21`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the feedback inhibitory loop.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    c_inh2pyr = NArray(
        label=r":math:`c_22`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the slow feedback inhibitory loop.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    c_pyr2inhf = NArray(
        label=r":math:`c_22`",
        default=numpy.array([0.3]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts from pyramidal to fast inhibitory.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    c_inh2inhf = NArray(
        label=r":math:`c_22`",
        default=numpy.array([0.1]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts from slow inhibitory to fast inhibitory.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    c_inhf2pyr = NArray(
        label=r":math:`c_22`",
        default=numpy.array([0.8]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts from fast inhibitory to pyramidals.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    p = NArray(
        label=r":math:`\p_{max}`",
        default=numpy.array([0.22]),
        domain=Range(lo=0.0, hi=0.22, step=0.01),
        doc="""Mean input firing rate""")

    sigma = NArray(
        label=r":math:`\sigma`",
        default=numpy.array([0.022]),
        domain=Range(lo=0.0, hi=0.05, step=0.005),
        doc="""Standard deviation of input firing rate following a Gaussian""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"vPyr": numpy.array([-1.0, 1.0]),
                 "vExc": numpy.array([-50.0, 50.0]),
                 "vInh": numpy.array([-5.0, 50.0]),
                 "vInhf": numpy.array([-5.0, 5.0]),
                 "vInhs2f": numpy.array([-5.0, 5.0]),

                 "xPyr": numpy.array([-6.0, 6.0]),
                 "xExc": numpy.array([-20.0, 20.0]),
                 "xInh": numpy.array([-50.0, 50.0]),
                 "xInhf": numpy.array([-50.0, 50.0]),
                 "xInhs2f": numpy.array([-50.0, 50.0])},

        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current 
        parameters, it is used as a mechanism for bounding random inital 
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("vPyr", "vExc", "vInh", "vInhf", "vInhs2f",  "xPyr", "xExc", "xInh",  "xInhf",  "xInhs2f"),
        default=("vPyr", "vExc", "vInh", "vInhf", "xPyr"),
        doc="""This represents the default state-variables of this Model to be
                                    monitored. It can be overridden for each Monitor if desired. The 
                                    corresponding state-variable indices for this model are :math:`y0 = 0`,
                                    :math:`y1 = 1`, :math:`y2 = 2`, :math:`y3 = 3`, :math:`y4 = 4`, and
                                    :math:`y5 = 5`""")

    state_variables = ["vPyr", "vExc", "vInh", "vInhf", "vInhs2f",  "xPyr", "xExc", "xInh",  "xInhf",  "xInhs2f"]
    _nvar = 10
    cvar = numpy.array([1, 2, 3], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The dynamic equations were taken from:
        TODO: add equations and finish the model ...
        """

        vPyr = state_variables[0, :]
        vExc = state_variables[1, :]
        vInh = state_variables[2, :]
        vInhf = state_variables[3, :]
        vInhs2f = state_variables[4, :]

        xPyr = state_variables[5, :]
        xExc = state_variables[6, :]
        xInh = state_variables[7, :]
        xInhf = state_variables[8, :]
        xInhs2f = state_variables[9, :]



        # NOTE: This is assumed to be \sum_j u_kj * S[y_{1_j} - y_{2_j}]
        lrc = coupling[0, :]
        src = local_coupling * (vExc - vInh)

        ## Intrinsic input
        input = numpy.random.normal(self.p, self.sigma, (len(coupling[0]), 1))

        S_pyr = self.e0 / (1 + numpy.exp(self.r * (self.v0 - (vExc - vInh - vInhf))))
        S_exc = self.e0 / (1 + numpy.exp(self.r * (self.v0 - self.c_pyr2exc * self.c * vPyr)))
        S_inh = self.e0 / (1 + numpy.exp(self.r * (self.v0 - self.c_pyr2inh * self.c * vPyr)))
        S_inhf = self.e0 / (1 + numpy.exp(self.r * (self.v0 - self.c_pyr2inhf * self.c * vPyr - self.c_inh2inhf * self.c * vInhs2f)))


        ## NOTE, for local couplings:
        # vExc, xExc excitatory interneurons - gate for intrinsic (p) and external (lrc, src) inputs.
        # vInh, xInh inhibitory interneurons; vInh_f, xInh_f fast interneurons
        # vPyr, xPyr pyramidal neurons

        dvPyr = xPyr
        dxPyr = self.He / self.tau_e * S_pyr - (2 * xPyr) / self.tau_e - vPyr / self.tau_e ** 2

        dvExc = xExc
        dxExc = self.He / self.tau_e * (input + self.c_exc2pyr * self.c * S_exc + lrc + src) - (2 * xExc) / self.tau_e - vExc / self.tau_e ** 2

        dvInh = xInh
        dxInh = self.Hi / self.tau_i * (self.c_inh2pyr * self.c * S_inh) - (2 * xInh) / self.tau_i - vInh / self.tau_i ** 2

        dvInhf = xInhf
        dxInhf = self.Hif / self.tau_if * (self.c_inhf2pyr * self.c * S_inhf) - (2 * xInhf) / self.tau_if - vInhf / self.tau_if ** 2

        dvInhs2f = xInhs2f
        dxInhs2f = self.Hi / self.tau_i * S_inh - (2 * xInh) / self.tau_i - vInh / self.tau_i ** 2

        derivative = numpy.array([dvPyr, dvExc, dvInh, dvInhf, dvInhs2f, dxPyr, dxExc, dxInh,  dxInhf,  dxInhs2f])

        return derivative


class JansenRitDavid2003(models.Model):
    """
    The Jansen and Rit model as studied by David et al., 2003
    They showed how an extension of Jansen-Rit could enhance simulation spectrum to look closer to MEG recordings.

    Their extension consisted on introducing a gamma oscillator coupled to a main alpha oscillator into each subnode.
    Couping the activity of those two oscillators resulted into broader alpha spectra, closer to reality.

    TODO:
    - Interregional coupling

    """

    # Define traited attributes for this model, these represent possible kwargs.
    He1 = NArray(
        label=":math:`He`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain in the first kinetic population.""")

    Hi1 = NArray(
        label=":math:`B`",
        default=numpy.array([22.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain in the first kinetic population.""")

    tau_e1 = NArray(
        label=":math:`a`",
        default=numpy.array([10.0]),
        domain=Range(lo=6.3, hi=20.0, step=0.1),
        doc="""Excitatory time constant (ms) in the first kinetic population.""")

    tau_i1 = NArray(
        label=":math:`b`",
        default=numpy.array([20.0]),
        domain=Range(lo=12.0, hi=40.0, step=0.2),
        doc="""Inhibitory time constant (ms) in the first kinetic population.""")

    He2 = NArray(
        label=":math:`He`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain in the second kinetic population.""")

    Hi2 = NArray(
        label=":math:`B`",
        default=numpy.array([22.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain in the second kinetic population.""")

    tau_e2 = NArray(
        label=":math:`a`",
        default=numpy.array([10.0]),
        domain=Range(lo=6.3, hi=20.0, step=0.1),
        doc="""Excitatory time constant (ms) in the second kinetic population""")

    tau_i2 = NArray(
        label=":math:`b`",
        default=numpy.array([20.0]),
        domain=Range(lo=12.0, hi=40.0, step=0.2),
        doc="""Inhibitory time constant (ms) in the second kinetic population.""")

    w = NArray(
        label=r":math:`w`",
        default=numpy.array([0.8]),
        domain=Range(lo=0., hi=1.0, step=0.05),
        doc="""Relative proportion of each kinectic population in the cortical area. 
        Multiplies population 1; (1-w)*population 2""")

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


    p = NArray(
        label=r":math:`\p_{mean}`",
        default=numpy.array([0.22]),
        domain=Range(lo=0.0, hi=0.32, step=0.01),
        doc="""Mean input firing rate""")

    sigma = NArray(
        label=r":math:`\sigma`",
        default=numpy.array([0.022]),
        domain=Range(lo=0.0, hi=0.05, step=0.005),
        doc="""Standard deviation of input firing rate following a Gaussian""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",

        default={"vPyr1": numpy.array([-1.0, 1.0]),
                 "vExc1": numpy.array([-500.0, 500.0]),
                 "vInh1": numpy.array([-50.0, 50.0]),
                 "xPyr1": numpy.array([-6.0, 6.0]),
                 "xExc1": numpy.array([-20.0, 20.0]),
                 "xInh1": numpy.array([-500.0, 500.0]),

                 "vPyr2": numpy.array([-1.0, 1.0]),
                 "vExc2": numpy.array([-500.0, 500.0]),
                 "vInh2": numpy.array([-50.0, 50.0]),
                 "xPyr2": numpy.array([-6.0, 6.0]),
                 "xExc2": numpy.array([-20.0, 20.0]),
                 "xInh2": numpy.array([-500.0, 500.0])},

        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current 
        parameters, it is used as a mechanism for bounding random inital 
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("vPyr1", "vExc1", "vInh1", "xPyr1", "xExc1", "xInh1",
                 "vPyr2", "vExc2", "vInh2", "xPyr2", "xExc2", "xInh2"),

        default=("vPyr1", "vExc1", "vInh1", "xPyr1",
                 "vPyr2", "vExc2", "vInh2", "xPyr2"),

        doc="""This represents the default state-variables of this Model to be
            monitored. It can be overridden for each Monitor if desired. 
            Correspondance in David 2003:
            vExc, xExc = v1, x1; vInh, xInh = v2, x2; vPyr, xPyr = v3, x3.
            Correspondance in Jansen-Rit 1995:
            vExc, xExc = y1, y4; vInh, xInh = y2, y5; vPyr, xPyr = y0, y3.""")

    state_variables = ["vPyr1", "vExc1", "vInh1", "xPyr1", "xExc1", "xInh1",
                       "vPyr2", "vExc2", "vInh2", "xPyr2", "xExc2", "xInh2"]
    _nvar = 12
    cvar = numpy.array([1, 2, 7, 8], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The dynamic equations were taken from:
        TODO: add equations and finish the model ...
        """

        vPyr1 = state_variables[0, :]
        vExc1 = state_variables[1, :]
        vInh1 = state_variables[2, :]
        xPyr1 = state_variables[3, :]
        xExc1 = state_variables[4, :]
        xInh1 = state_variables[5, :]

        vPyr2 = state_variables[6, :]
        vExc2 = state_variables[7, :]
        vInh2 = state_variables[8, :]
        xPyr2 = state_variables[9, :]
        xExc2 = state_variables[10, :]
        xInh2 = state_variables[11, :]

        sum_vPyr = self.w * vPyr1 + (1-self.w) * vPyr2
        sum_vExc_vInh = self.w * (vExc1 - vInh1) + (1-self.w) * (vExc2 - vInh2)

        # NOTE: This is assumed to be \sum_j u_kj * S[y_{1_j} - y_{2_j}]
        lrc = coupling[0, :]
        src = local_coupling * (sum_vExc_vInh)

        ## Intrinsic input
        input = numpy.random.normal(self.p, self.sigma, (len(coupling[0]), 1))

        S_pyr = self.e0 / (1 + numpy.exp(self.r * (self.v0 - sum_vExc_vInh)))
        S_exc = self.e0 / (1 + numpy.exp(self.r * (self.v0 - self.c_pyr2exc * self.c * sum_vPyr)))
        S_inh = self.e0 / (1 + numpy.exp(self.r * (self.v0 - self.c_pyr2inh * self.c * sum_vPyr)))

        ## NOTE, for local couplings:
        # vExc, xExc excitatory interneurons - gate for intrinsic (p) and external (lrc, src) inputs.
        # vInh, xInh inhibitory interneurons
        # vPyr, xPyr pyramidal neurons

        ### SLOW kinetic population
        dvPyr1 = xPyr1
        dvExc1 = xExc1
        dvInh1 = xInh1
        dxPyr1 = self.He1 / self.tau_e1 * S_pyr - (2 * xPyr1) / self.tau_e1 - vPyr1 / self.tau_e1 ** 2
        dxExc1 = self.He1 / self.tau_e1 * (input + self.c_exc2pyr * self.c * S_exc + lrc + src) - (2 * xExc1) / self.tau_e1 - vExc1 / self.tau_e1 ** 2
        dxInh1 = self.Hi1 / self.tau_i1 * (self.c_inh2pyr * self.c * S_inh) - (2 * xInh1) / self.tau_i1 - vInh1 / self.tau_i1 ** 2

        ### FAST kinetic population
        dvPyr2 = xPyr2
        dvExc2 = xExc2
        dvInh2 = xInh2
        dxPyr2 = self.He2 / self.tau_e2 * S_pyr - (2 * xPyr2) / self.tau_e2 - vPyr2 / self.tau_e2 ** 2
        dxExc2 = self.He2 / self.tau_e2 * (input + self.c_exc2pyr * self.c * S_exc + lrc + src) - (2 * xExc2) / self.tau_e2 - vExc2 / self.tau_e2 ** 2
        dxInh2 = self.Hi2 / self.tau_i2 * (self.c_inh2pyr * self.c * S_inh) - (2 * xInh2) / self.tau_i2 - vInh2 / self.tau_i2 ** 2

        derivative = numpy.array([dvPyr1, dvExc1, dvInh1, dxPyr1, dxExc1, dxInh1,
                                  dvPyr2, dvExc2, dvInh2, dxPyr2, dxExc2, dxInh2])

        return derivative


class JansenRitDavid2003_th(models.Model):
    """
    Extending the extension.

    Here, we propose to use the Jansen-Rit's model extension (David et al., 2003) to enhance spectral richness.
    And to combine David et al. (2005) hierarchical Jansen-Rit with the Thalamic relevance posed in Jones (2009).

    The hierarchical implementation allowed to give different weights to connections coming from different
    cortical layers. We will take advantage of that proposal but using the scheme to differentiate the
    inputs from cortico-cortical connections and the input from thalamo-cortical connections.

    David et al. didnt consider thalamo-cortical connections as: "they represent a minority of extrinsic connections:
    it is thought that at least 99% of axons in white matter link cortical areas of the same hemisphere."

    TODO:


    """

    # Define traited attributes for this model, these represent possible kwargs.
    He1 = NArray(
        label=":math:`He`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain in the first kinetic population.""")

    Hi1 = NArray(
        label=":math:`B`",
        default=numpy.array([22.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain in the first kinetic population.""")

    tau_e1 = NArray(
        label=":math:`a`",
        default=numpy.array([10.0]),
        domain=Range(lo=6.3, hi=20.0, step=0.1),
        doc="""Excitatory time constant (ms) in the first kinetic population.""")

    tau_i1 = NArray(
        label=":math:`b`",
        default=numpy.array([20.0]),
        domain=Range(lo=12.0, hi=40.0, step=0.2),
        doc="""Inhibitory time constant (ms) in the first kinetic population.""")

    He2 = NArray(
        label=":math:`He`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain in the second kinetic population.""")

    Hi2 = NArray(
        label=":math:`B`",
        default=numpy.array([22.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain in the second kinetic population.""")

    tau_e2 = NArray(
        label=":math:`a`",
        default=numpy.array([10.0]),
        domain=Range(lo=6.3, hi=20.0, step=0.1),
        doc="""Excitatory time constant (ms) in the second kinetic population""")

    tau_i2 = NArray(
        label=":math:`b`",
        default=numpy.array([20.0]),
        domain=Range(lo=12.0, hi=40.0, step=0.2),
        doc="""Inhibitory time constant (ms) in the second kinetic population.""")

    w = NArray(
        label=r":math:`w`",
        default=numpy.array([0.8]),
        domain=Range(lo=0., hi=1.0, step=0.05),
        doc="""Relative proportion of each kinectic population in the cortical area. 
        Multiplies population 1; (1-w)*population 2""")

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


    p = NArray(
        label=r":math:`\p_{mean}`",
        default=numpy.array([0.22]),
        domain=Range(lo=0.0, hi=0.32, step=0.01),
        doc="""Mean input firing rate""")

    sigma = NArray(
        label=r":math:`\sigma`",
        default=numpy.array([0.022]),
        domain=Range(lo=0.0, hi=0.05, step=0.005),
        doc="""Standard deviation of input firing rate following a Gaussian""")


    # k = NArray(
    #     label=r":math:`k1`",
    #     default=numpy.array([0.5]),
    #     domain=Range(lo=0.0, hi=1.0, step=0.05),
    #     doc="""Contribution of simulated areas on ROI signal. +
    #     In contrast, (1-k) contribution of unknown elements to signal (noise).""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        {
            "vExc1": numpy.array([-1.0, 1.0]),
            "xExc1": numpy.array([-2.0, 2.0]),
            "vInh1": numpy.array([-5.0, 5.0]),
            "xInh1": numpy.array([-5.0, 5.0]),
            "vPyr1": numpy.array([-1.0, 1.0]),
            "xPyr1": numpy.array([-6.0, 6.0]),

            "vExc2": numpy.array([-1.0, 1.0]),
            "xExc2": numpy.array([-2.0, 2.0]),
            "vInh2": numpy.array([-5.0, 5.0]),
            "xInh2": numpy.array([-5.0, 5.0]),
            "vPyr2": numpy.array([-1.0, 1.0]),
            "xPyr2": numpy.array([-6.0, 6.0]),
        },
        label="State Variable ranges [lo, hi]",
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current 
        parameters, it is used as a mechanism for bounding random inital 
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("vExc1", "xExc1", "vInh1", "xInh1", "vPyr1", "xPyr1",
                 "vExc2", "xExc2", "vInh2", "xInh2", "vPyr2", "xPyr2"),

        default=("vExc1", "vInh1", "vPyr1", "vExc2", "vInh2", "vPyr2"),

        doc="""This represents the default state-variables of this Model to be
            monitored. It can be overridden for each Monitor if desired. 
            Correspondance in David 2003:
            vExc, xExc = v1, x1; vInh, xInh = v2, x2; vPyr, xPyr = v3, x3.
            Correspondance in Jansen-Rit 1995:
            vExc, xExc = y1, y4; vInh, xInh = y2, y5; vPyr, xPyr = y0, y3.""")

    state_variables = ["vExc1", "xExc1", "vInh1", "xInh1", "vPyr1", "xPyr1",
                       "vExc2", "xExc2", "vInh2", "xInh2", "vPyr2", "xPyr2"]
    _nvar = 12
    cvar = numpy.array([0, 2, 6, 8], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The dynamic equations were taken from:
        TODO: add equations and finish the model ...
        """

        vExc1 = state_variables[0, :]
        xExc1 = state_variables[1, :]
        vInh1 = state_variables[2, :]
        xInh1 = state_variables[3, :]
        vPyr1 = state_variables[4, :]
        xPyr1 = state_variables[5, :]

        vExc2 = state_variables[6, :]
        xExc2 = state_variables[7, :]
        vInh2 = state_variables[8, :]
        xInh2 = state_variables[9, :]
        vPyr2 = state_variables[10, :]
        xPyr2 = state_variables[11, :]



        # NOTE for local couplings:
            # vExc, xExc excitatory interneurons - gate for intrinsic (p) and external (lrc, src) inputs.
            # vInh, xInh inhibitory interneurons; vPyr, xPyr pyramidal neurons
        sum_vPyr = self.w * vPyr1 + (1-self.w) * vPyr2
        sum_vExc_vInh = self.w * (vExc1 - vInh1) + (1-self.w) * (vExc2 - vInh2)


        # NOTE: This is assumed to be \sum_j u_kj * S[y_{1_j} - y_{2_j}]
        ## Weighted types of long range connections: [af - FeedForward - thalamo-cortical - proximal];
        # [ab - FeedBack - cortico-cortical - distal] Weights already applied in Coupling function.
        ab_lrc_fromcx = coupling[0, :]  # just cortico-cortical
        af_lrc_fromth = coupling[1, :]  # just thalamus

        src = local_coupling * (sum_vExc_vInh)

        ## Intrinsic input
        input = numpy.random.normal(self.p, self.sigma, (len(coupling[0]), 1))

        S_pyr = (self.e0) / (1 + numpy.exp(self.r * (self.v0 - sum_vExc_vInh)))
        S_exc = (self.c * self.c_exc2pyr * self.e0) / (1 + numpy.exp(self.r * (self.v0 - self.c * self.c_pyr2exc * sum_vPyr)))
        S_inh = (self.c * self.c_inh2pyr * self.e0) / (1 + numpy.exp(self.r * (self.v0 - self.c * self.c_pyr2inh * sum_vPyr)))


        ### SLOW kinetic population
        dvExc1 = xExc1
        dxExc1 = self.He1 / self.tau_e1 * (input + S_exc + src + af_lrc_fromth) - (2 * xExc1) / self.tau_e1 - (vExc1 / self.tau_e1**2)
        dvInh1 = xInh1
        dxInh1 = self.Hi1 / self.tau_i1 * (S_inh + ab_lrc_fromcx) - (2 * xInh1) / self.tau_i1 - (vInh1 / self.tau_i1**2)
        dvPyr1 = xPyr1
        dxPyr1 = self.He1 / self.tau_e1 * (S_pyr + ab_lrc_fromcx) - (2 * xPyr1) / self.tau_e1 - (vPyr1 / self.tau_e1**2)

        ### FAST kinetic population
        dvExc2 = xExc2
        dxExc2 = self.He2 / self.tau_e2 * (input + S_exc + src + af_lrc_fromth) - (2 * xExc2) / self.tau_e2 - (vExc2 / self.tau_e2**2)
        dvInh2 = xInh2
        dxInh2 = self.Hi2 / self.tau_i2 * (S_inh + ab_lrc_fromcx) - (2 * xInh2) / self.tau_i2 - (vInh2 / self.tau_i2**2)
        dvPyr2 = xPyr2
        dxPyr2 = self.He2 / self.tau_e2 * (S_pyr + ab_lrc_fromcx) - (2 * xPyr2) / self.tau_e2 - (vPyr2 / self.tau_e2**2)

        derivative = numpy.array([dvExc1, dxExc1, dvInh1, dxInh1, dvPyr1, dxPyr1,
                                  dvExc2, dxExc2, dvInh2, dxInh2, dvPyr2, dxPyr2])

        return derivative


class JansenRitDavid2005(models.Model):
    """

    Extension of the Jansen-Rit 1995 model to implement hierarchical connectivity
    of neocortical columns based on the work by David et al. (2005).

    """

    # Define traited attributes for this model, these represent possible kwargs.
    He = NArray(
        label=":math:`He`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain.""")

    Hi = NArray(
        label=":math:`B`",
        default=numpy.array([29.3]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain.""")

    tau_e = NArray(
        label=":math:`a`",
        default=numpy.array([10.0]),
        domain=Range(lo=6.3, hi=20.0, step=0.1),
        doc="""Excitatory time constant (ms)""")

    tau_i = NArray(
        label=":math:`b`",
        default=numpy.array([15.0]),
        domain=Range(lo=12.0, hi=40.0, step=0.2),
        doc="""Inhibitory time constant (ms)""")

    v0 = NArray(
        label=":math:`v_0`",
        default=numpy.array([0]),
        domain=Range(lo=3.12, hi=6.0, step=0.02),
        doc="""Firing threshold (PSP) for which a 50% firing rate is achieved.
        In other words, it is the value of the average membrane potential
        corresponding to the inflection point of the sigmoid [mV]; 
        6.0 in JansenRit1995 and DavidFriston2003; 0 in David2005""")

    e0 = NArray(
        label=":math:`e_0`",
        default=numpy.array([0.0025]),
        domain=Range(lo=0.00125, hi=0.00375, step=0.00001),
        doc="""Determines the maximum firing rate of the neural population
        [ms^-1]. In the papers its value is 2.5 [s^-1]; we convert to [ms^-1]""")

    r = NArray(
        label=":math:`r`",
        default=numpy.array([0.56]),
        domain=Range(lo=0.28, hi=0.84, step=0.01),
        doc="""Steepness of the sigmoidal transformation [mV^-1].""")


    gamma1_pyr2exc = NArray(
        label=r":math:`c_11`",
        default=numpy.array([50.0]),
        domain=Range(lo=0.5, hi=1.5, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback excitatory loop.
        It multiplies c; so c_11 = c = 135 contacts""")

    gamma2_exc2pyr = NArray(
        label=r":math:`c_12`",
        default=numpy.array([40.0]),
        domain=Range(lo=0.4, hi=1.2, step=0.1),
        doc="""Average probability of synaptic contacts in the slow feedback excitatory loop. 
        It multiplies c; so c_12 = c * 0.8 = 108 contacts""")

    gamma3_pyr2inh = NArray(
        label=r":math:`c_21`",
        default=numpy.array([12.0]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the feedback inhibitory loop.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    gamma4_inh2pyr = NArray(
        label=r":math:`c_22`",
        default=numpy.array([12.0]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the slow feedback inhibitory loop.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    p = NArray(
        label=r":math:`\p_{max}`",
        default=numpy.array([0.22]),
        domain=Range(lo=0.0, hi=0.22, step=0.01),
        doc="""Mean input firing rate""")

    sigma = NArray(
        label=r":math:`\sigma`",
        default=numpy.array([0.022]),
        domain=Range(lo=0.0, hi=0.05, step=0.005),
        doc="""Standard deviation of input firing rate following a Gaussian""")


    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"vExc": numpy.array([-1.0, 1.0]),
                 "vPyr_exc": numpy.array([-1.0, 1.0]),
                 "vPyr_inh": numpy.array([-5.0, 5.0]),
                 "xExc": numpy.array([-6.0, 6.0]),
                 "xPyr_exc": numpy.array([-2.0, 2.0]),
                 "xPyr_inh": numpy.array([-5.0, 5.0]),
                 "vInh": numpy.array([-5.0, 5.0]),
                 "xInh": numpy.array([-5.0, 5.0])},
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current 
        parameters, it is used as a mechanism for bounding random inital 
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("vExc", "vPyr_exc", "vPyr_inh", "xExc", "xPyr_exc", "xPyr_inh", "vInh", "xInh"),
        default=("vExc", "vPyr_exc", "vPyr_inh", "vInh",),
        doc="""This represents the default state-variables of this Model to be
                                    monitored. It can be overridden for each Monitor if desired. The 
                                    corresponding state-variable indices for this model are :math:`y0 = 0`,
                                    :math:`y1 = 1`, :math:`y2 = 2`, :math:`y3 = 3`, :math:`y4 = 4`, and
                                    :math:`y5 = 5`""")

    state_variables = ["vExc", "vPyr_exc", "vPyr_inh", "xExc", "xPyr_exc", "xPyr_inh", "vInh", "xInh"]
    _nvar = 8
    cvar = numpy.array([1, 2, 3], dtype=numpy.int32)



    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The dynamic equations were taken from:
        TODO: add equations and finish the model ...
        """

        # AF = 0.4  # 0.1
        # AB = 0.1  # 0.2
        # AL = 0  # 0.05

        vExc = state_variables[0, :]
        vPyr_exc = state_variables[1, :]
        vPyr_inh = state_variables[2, :]
        xExc = state_variables[3, :]
        xPyr_exc = state_variables[4, :]
        xPyr_inh   = state_variables[5, :]

        vInh = state_variables[6, :]
        xInh = state_variables[7, :]


        y = vPyr_exc - vPyr_inh


            ## Long-range coupling with hierarchical connections
        # delayed activity x1 - x2
        (lrcFF, lrcFB, lrcL) = coupling

        # Short-range coupling for surface simulations
        #  TODO how to model neighbour connections ¿lateral? ¿FF?
        srcL = (local_coupling * y) # * self.aL


        ## Intrinsic input
        input_u = numpy.random.normal(self.p, self.sigma, (len(coupling[0]), 1))


        # Afferences from population (i.e., S_pyr for the afferences of pyramidal cells to other subpops)
        S_pyr = (2 * self.e0) / (1 + numpy.exp(self.r * y)) - self.e0
        S_exc = (2 * self.e0) / (1 + numpy.exp(self.r * vExc)) - self.e0
        S_inh = (2 * self.e0) / (1 + numpy.exp(self.r * vInh)) - self.e0

        ## NOTE, for local couplings:
        # vExc, xExc excitatory interneurons - gate for intrinsic (p) and external (lrc, src) inputs.
        # vInh, xInh inhibitory interneurons
        # vPyr, xPyr, vPyr_aux, xPyr_aux pyramidal neurons in deep and superficial layers

        dvExc = xExc
        dvPyr_exc = xPyr_exc
        dvPyr_inh = xPyr_inh

        dxExc = self.He / self.tau_e * (input_u + lrcFF + lrcL + srcL + self.gamma1_pyr2exc * S_pyr) - (2 * xExc) / self.tau_e - vExc / self.tau_e ** 2
        dxPyr_exc = self.He / self.tau_e * (lrcFB + lrcL + srcL + self.gamma2_exc2pyr * S_exc) - (2 * xPyr_exc) / self.tau_e - vPyr_exc / self.tau_e ** 2
        dxPyr_inh = self.Hi / self.tau_i * (self.gamma4_inh2pyr * S_inh) - (2 * xPyr_inh) / self.tau_i - vPyr_inh / self.tau_i ** 2

        dvInh = xInh
        dxInh = self.He / self.tau_e * (lrcFB + lrcL + srcL + self.gamma3_pyr2inh * S_pyr) - (2 * xInh) / self.tau_e - vInh / self.tau_e ** 2

        derivative = numpy.array([dvExc,dvPyr_exc,  dvPyr_inh,  dxExc,dxPyr_exc, dxPyr_inh, dvInh, dxInh])

        return derivative


class DavidKiebel2008(models.Model):
    """

    Extension of the Jansen-Rit 1995 model to implement hierarchical connectivity
    of neocortical columns based on the work by David et al. (2005).
    Additional modification of the transfer function by Kiebel et al. (2008).

    This is the implementation of the NMM for erps, currently implemented in DCM @ 04/06/2024

    """

    # Define traited attributes for this model, these represent possible kwargs.
    He = NArray(
        label=":math:`He`",
        default=numpy.array([4]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain.""")

    Hi = NArray(
        label=":math:`B`",
        default=numpy.array([32]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain.""")

    tau_e = NArray(
        label=":math:`a`",
        default=numpy.array([8]),
        domain=Range(lo=6.3, hi=20.0, step=0.1),
        doc="""Excitatory time constant (ms)""")

    tau_i = NArray(
        label=":math:`b`",
        default=numpy.array([16.0]),
        domain=Range(lo=12.0, hi=40.0, step=0.2),
        doc="""Inhibitory time constant (ms)""")


    gamma1_pyr2exc = NArray(
        label=r":math:`c_11`",
        default=numpy.array([128.0]),
        domain=Range(lo=0.5, hi=150, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback excitatory loop.
        It multiplies c; so c_11 = c = 135 contacts""")

    gamma2_exc2pyr = NArray(
        label=r":math:`c_12`",
        default=numpy.array([102.0]),
        domain=Range(lo=0.4, hi=1.2, step=0.1),
        doc="""Average probability of synaptic contacts in the slow feedback excitatory loop. 
        It multiplies c; so c_12 = c * 0.8 = 108 contacts""")

    gamma3_pyr2inh = NArray(
        label=r":math:`c_21`",
        default=numpy.array([32.0]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the feedback inhibitory loop.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    gamma4_inh2pyr = NArray(
        label=r":math:`c_22`",
        default=numpy.array([32.0]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the slow feedback inhibitory loop.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")


    p = NArray(
        label=r":math:`\p_{max}`",
        default=numpy.array([0.22]),
        domain=Range(lo=0.0, hi=0.22, step=0.01),
        doc="""Mean input firing rate""")

    sigma = NArray(
        label=r":math:`\sigma`",
        default=numpy.array([0.022]),
        domain=Range(lo=0.0, hi=0.05, step=0.005),
        doc="""Standard deviation of input firing rate following a Gaussian""")


    rho1 = NArray(
        label=":math:`v_0`",
        default=numpy.array([0.67]),
        domain=Range(lo=0.312, hi=0.6, step=0.02),
        doc="""Slope of the sigmoid function. """)

    rho2 = NArray(
        label=":math:`e_0`",
        default=numpy.array([0.34]),
        domain=Range(lo=0.125, hi=0.5, step=0.00001),
        doc="""Translation of the sigmoid function.""")



    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"vExc": numpy.array([-1.0, 1.0]),
                 "vPyr_exc": numpy.array([-1.0, 1.0]),
                 "vPyr_inh": numpy.array([-5.0, 5.0]),
                 "xExc": numpy.array([-6.0, 6.0]),
                 "xPyr_exc": numpy.array([-2.0, 2.0]),
                 "xPyr_inh": numpy.array([-5.0, 5.0]),
                 "vInh": numpy.array([-5.0, 5.0]),
                 "xInh": numpy.array([-5.0, 5.0])},
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current 
        parameters, it is used as a mechanism for bounding random inital 
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("vExc", "vPyr_exc", "vPyr_inh", "xExc", "xPyr_exc", "xPyr_inh", "vInh", "xInh"),
        default=("vExc", "vPyr_exc", "vPyr_inh", "vInh",),
        doc="""This represents the default state-variables of this Model to be
                                    monitored. It can be overridden for each Monitor if desired. The 
                                    corresponding state-variable indices for this model are :math:`y0 = 0`,
                                    :math:`y1 = 1`, :math:`y2 = 2`, :math:`y3 = 3`, :math:`y4 = 4`, and
                                    :math:`y5 = 5`""")

    state_variables = ["vExc", "vPyr_exc", "vPyr_inh", "xExc", "xPyr_exc", "xPyr_inh", "vInh", "xInh"]
    _nvar = 8
    cvar = numpy.array([1, 2, 3], dtype=numpy.int32)



    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The dynamic equations were taken from:
        TODO: add equations and finish the model ...
        """

        vExc = state_variables[0, :]
        vPyr_exc = state_variables[1, :]
        vPyr_inh = state_variables[2, :]
        xExc = state_variables[3, :]
        xPyr_exc = state_variables[4, :]
        xPyr_inh   = state_variables[5, :]

        vInh = state_variables[6, :]
        xInh = state_variables[7, :]


        y = vPyr_exc - vPyr_inh


        ## External inputs.
        ## Long-range coupling with hierarchical connections
        # delayed activity x1 - x2
        (lrcFF, lrcFB, lrcL) = coupling

        # Short-range coupling for surface simulations. Considered as laterals.
        srcL = (local_coupling * y) # * self.aL

        ## Intrinsic input
        input_u = numpy.random.normal(self.p, self.sigma, (len(coupling[0]), 1))


        # Intra-column connections.
        # Afferences from population (i.e., S_pyr for the afferences of pyramidal cells to other subpops)
        # S_x = (1 / ( 1 + numpy.exp(-self.rho1 * (x - self.rho2)))) - (1 / (1 + numpy.exp(self.rho1 * self.rho2)))
        # From Kiebel et al. (2008); find it also in Moran et al. (2007).

        S_pyr = (1 / ( 1 + numpy.exp(-self.rho1 * (y - self.rho2)))) - (1 / (1 + numpy.exp(self.rho1 * self.rho2)))
        S_exc = (1 / ( 1 + numpy.exp(-self.rho1 * (vExc - self.rho2)))) - (1 / (1 + numpy.exp(self.rho1 * self.rho2)))
        S_inh = (1 / ( 1 + numpy.exp(-self.rho1 * (vInh - self.rho2)))) - (1 / (1 + numpy.exp(self.rho1 * self.rho2)))


        ## NOTE, for local couplings:
        # vExc, xExc excitatory interneurons - gate for intrinsic (p) and external (lrc, src) inputs.
        # vInh, xInh inhibitory interneurons
        # vPyr, xPyr, vPyr_aux, xPyr_aux pyramidal neurons in deep and superficial layers

        dvExc = xExc
        dvPyr_exc = xPyr_exc
        dvPyr_inh = xPyr_inh

        dxExc = self.He / self.tau_e * (lrcFF + lrcL + srcL + self.gamma1_pyr2exc * S_pyr + input_u) - (2 * xExc) / self.tau_e - vExc / self.tau_e ** 2
        dxPyr_exc = self.He / self.tau_e * (lrcFB + lrcL + srcL + self.gamma2_exc2pyr * S_exc) - (2 * xPyr_exc) / self.tau_e - vPyr_exc / self.tau_e ** 2
        dxPyr_inh = self.Hi / self.tau_i * (self.gamma4_inh2pyr * S_inh) - (2 * xPyr_inh) / self.tau_i - vPyr_inh / self.tau_i ** 2

        dvInh = xInh
        dxInh = self.He / self.tau_e * (lrcFB + lrcL + srcL + self.gamma3_pyr2inh * S_pyr) - (2 * xInh) / self.tau_e - vInh / self.tau_e ** 2

        derivative = numpy.array([dvExc,dvPyr_exc,  dvPyr_inh,  dxExc,dxPyr_exc, dxPyr_inh, dvInh, dxInh])

        return derivative



class myCanonicalMicroCircuit(models.Model):
    """
    This is my own interpretation and implementation of the CMC. It is not published anywhere @07/06/2024.

     Implementation of the Canonical Microcircuit based on the data gathered and modeled
     by Haesler and Maas (2006). Further discussions in Bastos (2012).

     Here, we implement the model as a neural mass model, prepared to be implemented
     in a surface simulation with lateral connections. We implement ideas from David (2005),
     from Wendling (2002), Pinotsis (2013) and from Jansen-Rit (1995) to develop the model.
     The main basis is David (2005).

     An implementation of the Canonical Microcircuit was proposed in Pinotsis (2013) as a
     mean field model. **However, the intrinsic conectivity patterns implemented there
     are not consistent with the scheme proposed by Haesler and Maas (2006). Pinotsis
     declared "By collapsing two pairs of cell types in the Haeusler and Maas model ... " but this
     collapse is not explained. They also include connections that does not correspond to the
     original scheme: 1) connections from deep pyramidals to inhibitory;  2) from superficial
     pyramidals to excitatory interneurons; 3) from inhibitory to excitatory interneurons.

    """


    # Define traited attributes for this model, these represent possible kwargs.
    He = NArray(
        label=":math:`He`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain.""")

    Hi_slow = NArray(
        label=":math:`B`",
        default=numpy.array([22.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of slow IPSP [mV] associated to dendritic projecting GABAergic interneurons""")

    Hi_fast = NArray(
        label=":math:`B`",
        default=numpy.array([10.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of the fast IPSP [mV] associated to somatic projecting GABAergic interneurons""")



    taue = NArray(
        label=":math:`a`",
        default=numpy.array([10.0]),
        domain=Range(lo=6.3, hi=20.0, step=0.1),
        doc="""Excitatory time constant (ms)""")

    taui_slow = NArray(
        label=":math:`b`",
        default=numpy.array([20.0]),
        domain=Range(lo=12.0, hi=40.0, step=0.2),
        doc="""Slow inhibitory time constant (ms) - dendritic GABAergic synapses""")

    taui_fast = NArray(
        label=":math:`b`",
        default=numpy.array([2.0]),
        domain=Range(lo=1.0, hi=40.0, step=0.2),
        doc="""Fast inhibitory time constant (ms) - somatic GABAergic synapses""")



    c_exc2pyrsup = NArray(
        label=r":math:`c_12`",
        default=numpy.array([135.0]),
        domain=Range(lo=0.4, hi=1.2, step=0.1),
        doc="""Average probability of synaptic contacts in the feedforward excitatory loop. 
        It multiplies c; so c_11 = c = 135 contacts""")


    c_pyrsup2pyrinf = NArray(
        label=r":math:`c_11`",
        default=numpy.array([108.0]),
        domain=Range(lo=0.5, hi=1.5, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback excitatory loop.
        It multiplies c; so c_12 = c * 0.8 = 108 contacts""")


    c_pyrsup2inhslow = NArray(
        label=r":math:`c_11`",
        default=numpy.array([33.75]),
        domain=Range(lo=0.5, hi=1.5, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback inhibitory loop.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    c_exc2inhfast = NArray(
        label=r":math:`c_21`",
        default=numpy.array([33.75]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the feedforward inhibitory loop.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    c_inhfast2pyrsup = NArray(
        label=r":math:`c_22`",
        default=numpy.array([33.75]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the slow feedback inhibitory loop.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    c_inhslow2pyrinf = NArray(
        label=r":math:`c_22`",
        default=numpy.array([33.75]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the slow feedback inhibitory loop.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")




    p = NArray(
        label=r":math:`\p_{max}`",
        default=numpy.array([0.22]),
        domain=Range(lo=0.0, hi=0.22, step=0.01),
        doc="""Mean input firing rate""")

    sigma = NArray(
        label=r":math:`\sigma`",
        default=numpy.array([0.022]),
        domain=Range(lo=0.0, hi=0.05, step=0.005),
        doc="""Standard deviation of input firing rate following a Gaussian""")



    v0 = NArray(
        label=":math:`v_0`",
        default=numpy.array([0]),
        domain=Range(lo=3.12, hi=6.0, step=0.02),
        doc="""Firing threshold (PSP) for which a 50% firing rate is achieved.
        In other words, it is the value of the average membrane potential
        corresponding to the inflection point of the sigmoid [mV]; 
        6.0 in JansenRit1995; 0 in David2005""")

    e0 = NArray(
        label=":math:`e_0`",
        default=numpy.array([0.0025]),
        domain=Range(lo=0.00125, hi=0.00375, step=0.00001),
        doc="""Determines the maximum firing rate of the neural population
        [ms^-1]. In the papers its value is 2.5 [s^-1]; we convert to [ms^-1]""")

    r = NArray(
        label=":math:`r`",
        default=numpy.array([0.56]),
        domain=Range(lo=0.28, hi=0.84, step=0.01),
        doc="""Steepness of the sigmoidal transformation [mV^-1].""")



    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={
            "vPyr_l2l3": numpy.array([-1.0, 1.0]),
            "vExc_l4": numpy.array([-1.0, 1.0]),
            "vPyr_l5l6": numpy.array([-5.0, 5.0]),
            "vInh_fast": numpy.array([-5.0, 5.0]),
            "vInh_slow": numpy.array([-5.0, 5.0]),

            "xPyr_l2l3": numpy.array([-1.0, 1.0]),
            "xExc_l4": numpy.array([-1.0, 1.0]),
            "xPyr_l5l6": numpy.array([-5.0, 5.0]),
            "xInh_fast": numpy.array([-5.0, 5.0]),
            "xInh_slow": numpy.array([-5.0, 5.0]),
    },
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current 
        parameters, it is used as a mechanism for bounding random inital 
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",

        choices=("vPyr_l2l3", "vExc_l4", "vPyr_l5l6", "vInh_fast", "vInh_slow",
                 "xPyr_l2l3", "xExc_l4", "xPyr_l5l6", "xInh_fast", "xInh_slow"),

        default=("vPyr_l2l3", "vExc_l4", "vPyr_l5l6", "vInh_fast", "vInh_slow", ),
        doc="""This represents the default state-variables of this Model to be
                                    monitored. It can be overridden for each Monitor if desired. The 
                                    corresponding state-variable indices for this model are :math:`y0 = 0`,
                                    :math:`y1 = 1`, :math:`y2 = 2`, :math:`y3 = 3`, :math:`y4 = 4`, and
                                    :math:`y5 = 5`""")

    state_variables = ["vPyr_l2l3", "vExc_l4", "vPyr_l5l6", "vInh_fast", "vInh_slow",
                       "xPyr_l2l3", "xExc_l4", "xPyr_l5l6", "xInh_fast", "xInh_slow"]
    _nvar = 10
    cvar = numpy.array([0, 2], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The dynamic equations were taken from:
        TODO: add equations and finish the model ...
        """

        vPyr_l2l3 = state_variables[0, :]
        vExc_l4   = state_variables[1, :]
        vPyr_l5l6 = state_variables[2, :]
        vInh_fast = state_variables[3, :]
        vInh_slow = state_variables[4, :]

        xPyr_l2l3 = state_variables[5, :]
        xExc_l4   = state_variables[6, :]
        xPyr_l5l6 = state_variables[7, :]
        xInh_fast = state_variables[8, :]
        xInh_slow = state_variables[9, :]


        ## Long-range coupling with hierarchical connections
        # delayed activity x1 - x2
        lrcFF, lrcFB = coupling[:2]

        # Short-range coupling for surface simulations: lateral connections
        # as proposed in pinotsis (2013) each layer connects horizontally.
        srcL = (local_coupling * vPyr_l5l6) # * self.aL


        ## Intrinsic input
        input_u = numpy.random.normal(self.p, self.sigma, (len(coupling[0]), 1))

        # Afferences to subpopulations in FR.
        S_pyrsup = (2 * self.e0) / (1 + numpy.exp(self.r * (self.c_exc2pyrsup * vExc_l4 - self.c_inhfast2pyrsup * vInh_fast))) - self.e0
        S_pyrinf = (2 * self.e0) / (1 + numpy.exp(self.r * (self.c_pyrsup2pyrinf * vPyr_l2l3 - self.c_inhslow2pyrinf * vInh_slow))) - self.e0
        S_inhfast = (2 * self.e0) / (1 + numpy.exp(self.r * self.c_exc2inhfast * vExc_l4)) - self.e0
        S_inhslow = (2 * self.e0) / (1 + numpy.exp(self.r * self.c_pyrsup2inhslow * vPyr_l2l3)) - self.e0



        ## NOTE, for local couplings:
        # vExc, xExc excitatory interneurons - gate for intrinsic (p) and external (lrc, src) inputs.
        # vInh, xInh inhibitory interneurons
        # vPyr, xPyr, vPyr_aux, xPyr_aux pyramidal neurons in deep and superficial layers

        vPyr_l2l3 = xPyr_l2l3
        vExc_l4   = xExc_l4
        vPyr_l5l6 = xPyr_l5l6
        vInh_fast = xInh_fast
        vInh_slow = xInh_slow

        xPyr_l2l3 = self.He / self.taue * (lrcFB + srcL + S_pyrsup) - (2 * xPyr_l2l3) / self.taue - vPyr_l2l3 / self.taue ** 2
        xExc_l4   = self.He / self.taue * (input_u + lrcFF + srcL) - (2 * xExc_l4) / self.taue - vExc_l4 / self.taue ** 2
        xPyr_l5l6 = self.He / self.taue * (lrcFB + srcL + S_pyrinf) - (2 * xPyr_l5l6) / self.taue - vPyr_l5l6 / self.taue ** 2
        xInh_fast = self.Hi_fast / self.taui_fast * (lrcFB + srcL + S_inhfast) - (2 * xInh_fast) / self.taui_fast - vInh_fast / self.taui_fast ** 2
        xInh_slow = self.Hi_slow / self.taui_slow * (lrcFB + srcL + S_inhslow) - (2 * xInh_slow) / self.taui_slow - vInh_slow / self.taui_slow ** 2


        derivative = numpy.array([vPyr_l2l3, vExc_l4, vPyr_l5l6, vInh_fast, vInh_slow,
                                  xPyr_l2l3, xExc_l4, xPyr_l5l6, xInh_fast, xInh_slow])

        return derivative




