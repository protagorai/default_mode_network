"""Benchmark the vectorized fast simulation."""
import time
from sdmn.networks.celegans import build_connectome_network
from sdmn.visualization.fast_sim import _prepare_arrays, _fast_step, build_frame_from_arrays

net = build_connectome_network()
arrays = _prepare_arrays(net)
n_syn = len(arrays["syn_pre"])
n_gap = len(arrays["gap_a"])
print(f"Network: {arrays['n']} neurons, {n_syn} synapses, {n_gap} gaps")

N = 10000
t0 = time.perf_counter()
for _ in range(N):
    _fast_step(arrays, 0.1)
elapsed = time.perf_counter() - t0

sim_ms = N * 0.1
print(f"{N} steps in {elapsed:.3f}s = {N/elapsed:.0f} steps/s")
print(f"Sim time: {sim_ms:.0f} ms in {elapsed:.3f}s = {sim_ms/elapsed/1000:.1f}x realtime")

frame = build_frame_from_arrays(arrays, sim_ms, N)
bri = [(v + 65) / 25 for v in frame.voltages.values()]
print(f"Brightness: min={min(bri):.2f} max={max(bri):.2f} mean={sum(bri)/len(bri):.3f}")
print(f"Active synapses: {len(frame.active_synapses)}")
