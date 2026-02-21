# Scenario analysis (EV + storage impact)

import pandapower as pp
import numpy as np
import matplotlib.pyplot as plt

def create_network():

    net = pp.create_empty_network()

    for i in range(33):
        pp.create_bus(net, vn_kv=12.66)

    for i in range(1, 33):
        pp.create_line_from_parameters(
            net, i-1, i,
            length_km=1,
            r_ohm_per_km=0.1,
            x_ohm_per_km=0.05,
            c_nf_per_km=0,
            max_i_ka=1
        )

    for i in range(1, 33):
        pp.create_load(net, bus=i, p_mw=0.1, q_mvar=0.05)

    pp.create_ext_grid(net, bus=0)

    return net


def run_case(ev=False, storage=False):

    net = create_network()

    if ev:
        for b in [10, 20, 30]:
            pp.create_load(net, bus=b, p_mw=0.3)

    if storage:
        for b in [15, 25]:
            pp.create_storage(net, bus=b, p_mw=-0.2, max_e_mwh=0.5)

    pp.runpp(net)

    return net


cases = {
    "Base": run_case(),
    "EV": run_case(ev=True),
    "EV+Storage": run_case(ev=True, storage=True),
}

# Voltage plot
plt.figure()

for name, net in cases.items():
    plt.plot(net.res_bus.vm_pu, label=name)

plt.legend()
plt.xlabel("Bus")
plt.ylabel("Voltage")
plt.title("Voltage Profile")

plt.savefig("results/figures/voltage_profiles.png", dpi=300)
plt.show()

# Loss comparison
losses = [net.res_line.pl_mw.sum() for net in cases.values()]

plt.figure()
plt.bar(cases.keys(), losses)
plt.ylabel("Loss (MW)")
plt.title("Loss Comparison")

plt.savefig("results/figures/loss_comparison.png", dpi=300)
plt.show()