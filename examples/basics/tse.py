from epgpy import operators, functions, plotting


# parameters
FA = 120
ESP = 10
Nrf = 10

# operators
exc = operators.T(90, 90)
rfc = operators.T(FA, 0)
shift = operators.S(1, duration=ESP / 2)
adc = operators.ADC

seq = [exc] + [[shift, rfc, shift, adc]] * Nrf


# signal = functions.simulate(seq)

fig = plotting.plot_epg(seq, title="Turbo spin echo sequence")
plotting.show()
