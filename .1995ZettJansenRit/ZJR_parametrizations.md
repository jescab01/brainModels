Parametrizations for Zetterberg-Jansen-Rit (1995) model
---

Forrester (2019) - ok - Raising too large values. Not working.

Parameters for Jansen Rit model. Zetterberg is not a clean generalization of JansenRit as
there are some coupling parameters that are not acting in the same way. In JR, some of them act before Sigmoid,
in Zetterberg, coupling always act before rate2voltage operator. 

    m = models.ZetterbergJansen(He=np.array([7]), Hi=np.array([22]),

                    P=np.array([20]), Q=np.array([0]), U=np.array([0]), # P originally at 120

                    e0=np.array([2.5]),

                    gamma_1=np.array([135]), gamma_1T=np.array([0]), gamma_2=np.array([108]), gamma_2T=np.array([1]),
                    gamma_3=np.array([33.75]), gamma_3T=np.array([0]), gamma_4=np.array([33.75]), gamma_5=np.array([0]),

                    ke=np.array([100]), ki=np.array([50]),

                    rho_1=np.array([0.56]), rho_2=np.array([6]))

Stefanovski (2019) - ok - Not working. Based on Spiegler (2010) bifurcation analysis.

    m = models.ZetterbergJansen(He=np.array([3.25]), Hi=np.array([22]),

                    P=np.array([0.1085]), Q=np.array([0]), U=np.array([0]),

                    e0=np.array([2.5]),

                    gamma_1=np.array([135]), gamma_1T=np.array([0]), gamma_2=np.array([108]), gamma_2T=np.array([1]),
                    gamma_3=np.array([33.75]), gamma_3T=np.array([0]), gamma_4=np.array([33.75]), gamma_5=np.array([0]),

                    ke=np.array([0.1]), ki=np.array([0.05]),

                    rho_1=np.array([0.56]), rho_2=np.array([6]))

Sanz-Leon (2015) 

    m = models.ZetterbergJansen(He=np.array([3.25]), Hi=np.array([22]),

                    P=np.array([0.12]), Q=np.array([0.12]), U=np.array([0.12]),

                    e0=np.array([0.0025]),

                    gamma_1=np.array([135]), gamma_1T=np.array([1]), gamma_2=np.array([108]), gamma_2T=np.array([1]),
                    gamma_3=np.array([33.75]), gamma_3T=np.array([1]), gamma_4=np.array([33.75]), gamma_5=np.array([15]),

                    ke=np.array([0.1]), ki=np.array([0.05]),

                    rho_1=np.array([0.56]), rho_2=np.array([6]))