from epgpy import operators, plotting

# parameters
FA = 30
TR = 10
Nrf = 15

# operators
rf = operators.T(FA, 0)
shift1 = operators.S(-1, duration=TR / 3)
rx1 = operators.E(TR / 3, 1e3, 1e2)
shift2 = operators.S(2, duration=TR * 2 / 3)
rx2 = operators.E(TR * 2 / 3, 1e3, 1e2)

seq = [[rf, shift1, rx1, shift2, rx2]] * Nrf


plotting.plot_epg(seq, title="SSFP sequence")
plotting.show()
