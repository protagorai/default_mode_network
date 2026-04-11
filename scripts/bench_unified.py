"""Benchmark the unified vectorized engine with full ion channels."""
import time
import numpy as np
from sdmn.networks.celegans import build_connectome_network, SimulationConfig, PlasticityConfig
from sdmn.visualization.fast_sim import _prepare_arrays, _fast_step, build_frame_from_arrays

net = build_connectome_network()
net.config = SimulationConfig(
    dt=0.1,
    plasticity=PlasticityConfig(enable_stdp=True, enable_habituation=True),
)
arrays = _prepare_arrays(net)

print(f"Network: {arrays['n']} neurons, {len(arrays['syn_pre'])} synapses, {len(arrays['gap_a'])} gaps")
print(f"Ion channels: Ca/K/KCa per neuron, STDP+habituation enabled")

arrays["I_ext"][arrays["id_to_idx"]["AWCL"]] = 60.0

N = 10000
t0 = time.perf_counter()
for _ in range(N):
    _fast_step(arrays, 0.1)
elapsed = time.perf_counter() - t0

sim_ms = N * 0.1
print(f"\n{N} steps in {elapsed:.3f}s = {N/elapsed:.0f} steps/s")
print(f"Sim time: {sim_ms:.0f} ms in {elapsed:.3f}s = {sim_ms/elapsed/1000:.1f}x realtime")

frame = build_frame_from_arrays(arrays, sim_ms, N)
vs = list(frame.voltages.values())
print(f"\nVoltage: min={min(vs):.1f}  max={max(vs):.1f}  mean={np.mean(vs):.1f} mV")
print(f"Active synapses: {len(frame.active_synapses)}")
print(f"Active gaps: {len(frame.active_gaps)}")

ca = arrays["Ca_internal"]
print(f"Ca internal: min={ca.min():.1f}  max={ca.max():.1f}  mean={ca.mean():.1f} nM")
print(f"m_Ca: min={arrays['m_Ca'].min():.3f}  max={arrays['m_Ca'].max():.3f}")
print(f"m_K:  min={arrays['m_K'].min():.3f}  max={arrays['m_K'].max():.3f}")

from sdmn.visualization.fast_sim import get_plasticity_stats
stats = get_plasticity_stats(arrays)
print(f"\nPlasticity: strengthened={stats['n_strengthened']}  weakened={stats['n_weakened']}  habituated={stats['n_habituated']}")
print(f"Efficacy: mean={stats['efficacy_mean']:.3f}  min={stats['efficacy_min']:.3f}")
