"""
Unified vectorized simulation engine.

Reproduces the full biophysical model from ``CElegansNeuron`` (Ca/K/KCa ion
channels, calcium dynamics) and ``GradedChemicalSynapse`` (dual-exponential
kinetics, reversal-potential current, per-synapse parameters) on flat NumPy
arrays for 10-20x speedup over the per-object Python path.

Also implements STDP learning, short-term habituation, and synaptic delay
-- all reading configuration from ``PlasticityConfig`` on the network.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from sdmn.networks.celegans.network_manager import CElegansNetwork, SimulationFrame

FARADAY = 96485.0


def _sigmoid(V, V_half, k):
    """Vectorized Boltzmann sigmoid."""
    return 1.0 / (1.0 + np.exp(-(V - V_half) / k))


def _tau(V, tau_min, tau_max, V_half_tau, k_tau):
    """Vectorized voltage-dependent time constant."""
    return np.maximum(0.001, tau_min + (tau_max - tau_min) / (1.0 + np.exp((V - V_half_tau) / k_tau)))


def _prepare_arrays(network: CElegansNetwork):
    """Extract full network structure into flat arrays."""
    ids = sorted(network.neurons.keys())
    n = len(ids)
    id_to_idx = {nid: i for i, nid in enumerate(ids)}

    V = np.empty(n)
    g_leak = np.empty(n)
    E_leak = np.empty(n)
    C_m = np.empty(n)

    # Ion channel parameters (per-neuron)
    g_Ca = np.empty(n);  E_Ca = np.empty(n)
    V_half_Ca = np.empty(n); k_Ca = np.empty(n)
    tau_Ca_min = np.empty(n); tau_Ca_max = np.empty(n)
    V_half_tau_Ca = np.empty(n); k_tau_Ca = np.empty(n)

    g_K = np.empty(n); E_K = np.empty(n)
    V_half_K = np.empty(n); k_K = np.empty(n)
    tau_K_min = np.empty(n); tau_K_max = np.empty(n)
    V_half_tau_K = np.empty(n); k_tau_K = np.empty(n)

    g_KCa = np.empty(n); Ca_half = np.empty(n); tau_KCa = np.empty(n)
    Ca_rest = np.empty(n); tau_Ca_removal = np.empty(n)
    f_Ca = np.empty(n); cell_volume = np.empty(n)

    m_Ca = np.empty(n); m_K = np.empty(n); m_KCa = np.zeros(n)
    Ca_internal = np.empty(n)

    neuron_classes = {}

    for i, nid in enumerate(ids):
        neuron = network.neurons[nid]
        p = neuron.params
        V[i] = neuron.voltage
        g_leak[i] = p.g_leak; E_leak[i] = p.E_leak; C_m[i] = p.C_m

        if hasattr(neuron, "neuron_class"):
            neuron_classes[nid] = neuron.neuron_class.value
        else:
            neuron_classes[nid] = "unknown"

        cp = getattr(neuron, "celegans_params", None)
        if cp is not None:
            g_Ca[i] = cp.g_Ca; E_Ca[i] = cp.E_Ca
            V_half_Ca[i] = cp.V_half_Ca; k_Ca[i] = cp.k_Ca
            tau_Ca_min[i] = cp.tau_Ca_min; tau_Ca_max[i] = cp.tau_Ca_max
            V_half_tau_Ca[i] = cp.V_half_tau_Ca; k_tau_Ca[i] = cp.k_tau_Ca
            g_K[i] = cp.g_K; E_K[i] = cp.E_K
            V_half_K[i] = cp.V_half_K; k_K[i] = cp.k_K
            tau_K_min[i] = cp.tau_K_min; tau_K_max[i] = cp.tau_K_max
            V_half_tau_K[i] = cp.V_half_tau_K; k_tau_K[i] = cp.k_tau_K
            g_KCa[i] = cp.g_KCa; Ca_half[i] = cp.Ca_half; tau_KCa[i] = cp.tau_KCa
            Ca_rest[i] = cp.Ca_rest; tau_Ca_removal[i] = cp.tau_Ca_removal
            f_Ca[i] = cp.f_Ca; cell_volume[i] = cp.cell_volume
            m_Ca[i] = getattr(neuron, "m_Ca", _sigmoid(V[i], cp.V_half_Ca, cp.k_Ca))
            m_K[i] = getattr(neuron, "m_K", _sigmoid(V[i], cp.V_half_K, cp.k_K))
            m_KCa[i] = getattr(neuron, "m_KCa", 0.0)
            Ca_internal[i] = getattr(neuron, "Ca_internal", cp.Ca_rest)
        else:
            g_Ca[i] = 0; E_Ca[i] = 50; V_half_Ca[i] = -20; k_Ca[i] = 5
            tau_Ca_min[i] = 0.5; tau_Ca_max[i] = 5; V_half_tau_Ca[i] = -30; k_tau_Ca[i] = 10
            g_K[i] = 0; E_K[i] = -80; V_half_K[i] = -25; k_K[i] = 10
            tau_K_min[i] = 1; tau_K_max[i] = 10; V_half_tau_K[i] = -30; k_tau_K[i] = 10
            g_KCa[i] = 0; Ca_half[i] = 100; tau_KCa[i] = 50
            Ca_rest[i] = 50; tau_Ca_removal[i] = 100; f_Ca[i] = 0.01; cell_volume[i] = 1.0
            m_Ca[i] = 0; m_K[i] = 0; m_KCa[i] = 0; Ca_internal[i] = 50

    # Synapse arrays
    syn_pre_l, syn_post_l, syn_weight_l = [], [], []
    syn_V_thresh_l, syn_k_release_l = [], []
    syn_tau_rise_l, syn_tau_decay_l = [], []
    syn_E_syn_l, syn_exc_l = [], []

    for syn in network.chemical_synapses.values():
        pi = id_to_idx.get(syn.presynaptic_neuron_id)
        qi = id_to_idx.get(syn.postsynaptic_neuron_id)
        if pi is not None and qi is not None:
            sp = syn.graded_params
            syn_pre_l.append(pi); syn_post_l.append(qi)
            syn_weight_l.append(sp.weight)
            syn_V_thresh_l.append(sp.V_thresh)
            syn_k_release_l.append(sp.k_release)
            syn_tau_rise_l.append(sp.tau_rise)
            syn_tau_decay_l.append(sp.tau_decay)
            syn_E_syn_l.append(sp.E_syn)
            syn_exc_l.append(1.0 if sp.synapse_type.value == "excitatory" else -1.0)

    n_syn = len(syn_pre_l)
    syn_pre = np.array(syn_pre_l, dtype=np.int32)
    syn_post = np.array(syn_post_l, dtype=np.int32)
    syn_weight = np.array(syn_weight_l)
    syn_weight_initial = syn_weight.copy()
    syn_V_thresh = np.array(syn_V_thresh_l)
    syn_k_release = np.array(syn_k_release_l)
    syn_tau_rise = np.array(syn_tau_rise_l)
    syn_tau_decay = np.array(syn_tau_decay_l)
    syn_E_syn = np.array(syn_E_syn_l)
    syn_exc = np.array(syn_exc_l)
    syn_g_rise = np.zeros(n_syn)
    syn_g_decay = np.zeros(n_syn)

    # Gap junction arrays
    gap_a_l, gap_b_l, gap_g_l = [], [], []
    for gap in network.gap_junctions.values():
        ai = id_to_idx.get(gap.presynaptic_neuron_id)
        bi = id_to_idx.get(gap.postsynaptic_neuron_id)
        if ai is not None and bi is not None:
            gap_a_l.append(ai); gap_b_l.append(bi)
            gap_g_l.append(gap.gap_params.conductance)

    gap_a = np.array(gap_a_l, dtype=np.int32)
    gap_b = np.array(gap_b_l, dtype=np.int32)
    gap_g = np.array(gap_g_l) if gap_g_l else np.array([], dtype=np.float64)

    pc = network.config.plasticity

    return {
        "ids": ids, "id_to_idx": id_to_idx, "n": n,
        "V": V, "g_leak": g_leak, "E_leak": E_leak, "C_m": C_m,
        "I_ext": np.zeros(n),
        # Ion channels (per-neuron)
        "g_Ca": g_Ca, "E_Ca": E_Ca, "V_half_Ca": V_half_Ca, "k_Ca": k_Ca,
        "tau_Ca_min": tau_Ca_min, "tau_Ca_max": tau_Ca_max,
        "V_half_tau_Ca": V_half_tau_Ca, "k_tau_Ca": k_tau_Ca,
        "g_K": g_K, "E_K": E_K, "V_half_K": V_half_K, "k_K": k_K,
        "tau_K_min": tau_K_min, "tau_K_max": tau_K_max,
        "V_half_tau_K": V_half_tau_K, "k_tau_K": k_tau_K,
        "g_KCa": g_KCa, "Ca_half": Ca_half, "tau_KCa": tau_KCa,
        "Ca_rest": Ca_rest, "tau_Ca_removal": tau_Ca_removal,
        "f_Ca": f_Ca, "cell_volume": cell_volume,
        "m_Ca": m_Ca, "m_K": m_K, "m_KCa": m_KCa, "Ca_internal": Ca_internal,
        # Synapses
        "syn_pre": syn_pre, "syn_post": syn_post,
        "syn_weight": syn_weight, "syn_weight_initial": syn_weight_initial,
        "syn_V_thresh": syn_V_thresh, "syn_k_release": syn_k_release,
        "syn_tau_rise": syn_tau_rise, "syn_tau_decay": syn_tau_decay,
        "syn_E_syn": syn_E_syn, "syn_exc": syn_exc,
        "syn_g_rise": syn_g_rise, "syn_g_decay": syn_g_decay,
        "syn_efficacy": np.ones(n_syn),
        # Gap junctions
        "gap_a": gap_a, "gap_b": gap_b, "gap_g": gap_g,
        # STDP
        "trace_pre": np.zeros(n), "trace_post": np.zeros(n),
        "stdp_accumulator": 0,
        # Config
        "neuron_classes": neuron_classes,
        "plasticity_config": pc,
        "dmn_config": network.config.dmn,
    }


def _fast_step(arrays: dict, dt: float) -> None:
    """Full vectorized step: ion channels + synaptic kinetics + plasticity."""
    V = arrays["V"]
    n = arrays["n"]
    pc = arrays["plasticity_config"]

    # ---- Ion channel gating (vectorized first-order kinetics) ----
    m_Ca_inf = _sigmoid(V, arrays["V_half_Ca"], arrays["k_Ca"])
    t_Ca = _tau(V, arrays["tau_Ca_min"], arrays["tau_Ca_max"],
                arrays["V_half_tau_Ca"], arrays["k_tau_Ca"])
    arrays["m_Ca"] += dt * (m_Ca_inf - arrays["m_Ca"]) / t_Ca
    np.clip(arrays["m_Ca"], 0, 1, out=arrays["m_Ca"])

    m_K_inf = _sigmoid(V, arrays["V_half_K"], arrays["k_K"])
    t_K = _tau(V, arrays["tau_K_min"], arrays["tau_K_max"],
               arrays["V_half_tau_K"], arrays["k_tau_K"])
    arrays["m_K"] += dt * (m_K_inf - arrays["m_K"]) / t_K
    np.clip(arrays["m_K"], 0, 1, out=arrays["m_K"])

    m_KCa_inf = arrays["Ca_internal"] / (arrays["Ca_internal"] + arrays["Ca_half"])
    arrays["m_KCa"] += dt * (m_KCa_inf - arrays["m_KCa"]) / arrays["tau_KCa"]
    np.clip(arrays["m_KCa"], 0, 1, out=arrays["m_KCa"])

    # ---- Ionic currents ----
    I_leak = arrays["g_leak"] * (V - arrays["E_leak"])
    I_Ca = arrays["g_Ca"] * (arrays["m_Ca"] ** 2) * (V - arrays["E_Ca"])
    I_K = arrays["g_K"] * (arrays["m_K"] ** 4) * (V - arrays["E_K"])
    I_KCa = arrays["g_KCa"] * arrays["m_KCa"] * (V - arrays["E_K"])
    I_ion = -I_Ca - I_K - I_KCa  # sign: positive = depolarizing

    # ---- Calcium dynamics ----
    Ca_influx = -arrays["f_Ca"] * I_Ca / (2.0 * FARADAY * arrays["cell_volume"])
    Ca_removal = (arrays["Ca_internal"] - arrays["Ca_rest"]) / arrays["tau_Ca_removal"]
    arrays["Ca_internal"] += dt * (Ca_influx - Ca_removal)
    np.clip(arrays["Ca_internal"], 0, 10000, out=arrays["Ca_internal"])

    # ---- Synaptic current (dual-exponential kinetics) ----
    I_syn = np.zeros(n)
    if len(arrays["syn_pre"]) > 0:
        V_pre = V[arrays["syn_pre"]]
        release = _sigmoid(V_pre, arrays["syn_V_thresh"], arrays["syn_k_release"])
        eff_weight = arrays["syn_weight"] * arrays["syn_efficacy"]

        # Dual-exponential conductance update
        arrays["syn_g_rise"] *= np.exp(-dt / np.maximum(arrays["syn_tau_rise"], 0.01))
        arrays["syn_g_decay"] *= np.exp(-dt / np.maximum(arrays["syn_tau_decay"], 0.01))
        increment = release * eff_weight
        arrays["syn_g_rise"] += increment
        arrays["syn_g_decay"] += increment
        g_syn = np.maximum(0, arrays["syn_g_decay"] - arrays["syn_g_rise"])

        # Conductance-based current: I = g * (V_post - E_syn)
        V_post = V[arrays["syn_post"]]
        I_per_syn = g_syn * (V_post - arrays["syn_E_syn"]) * arrays["syn_exc"]
        np.add.at(I_syn, arrays["syn_post"], -I_per_syn)

        # Habituation
        if pc.enable_habituation:
            arrays["syn_efficacy"] -= release * pc.habituation_depression_rate
            arrays["syn_efficacy"] += (1.0 - arrays["syn_efficacy"]) * pc.habituation_recovery_rate
            np.clip(arrays["syn_efficacy"], pc.habituation_floor, 1.0,
                    out=arrays["syn_efficacy"])

    # ---- Gap junction current ----
    I_gap = np.zeros(n)
    if len(arrays["gap_a"]) > 0:
        dV = V[arrays["gap_b"]] - V[arrays["gap_a"]]
        I_pair = arrays["gap_g"] * dV
        np.add.at(I_gap, arrays["gap_a"], I_pair)
        np.add.at(I_gap, arrays["gap_b"], -I_pair)

    # ---- Voltage integration (Euler with adaptive clamp) ----
    dVdt = (-I_leak + I_ion + I_syn + I_gap + arrays["I_ext"]) / arrays["C_m"]

    # Limit voltage change per step for Euler stability
    dV = dt * dVdt
    np.clip(dV, -5.0, 5.0, out=dV)
    arrays["V"] += dV
    np.clip(arrays["V"], -90, 40, out=arrays["V"])

    # ---- STDP ----
    if pc.enable_stdp:
        activation = np.clip((V - (-65.0)) / 25.0, 0.0, 1.0)
        decay = np.exp(-dt / pc.stdp_tau_trace)
        arrays["trace_pre"] = arrays["trace_pre"] * decay + activation
        arrays["trace_post"] = arrays["trace_post"] * decay + activation

        arrays["stdp_accumulator"] += 1
        if arrays["stdp_accumulator"] >= pc.stdp_update_interval and len(arrays["syn_pre"]) > 0:
            arrays["stdp_accumulator"] = 0
            pre_t = arrays["trace_pre"][arrays["syn_pre"]]
            post_t = arrays["trace_post"][arrays["syn_post"]]
            dw = pc.stdp_learning_rate * (pre_t * post_t - 0.5 * post_t * (1.0 - pre_t))
            arrays["syn_weight"] += dw
            w_max = arrays["syn_weight_initial"] * pc.stdp_w_max_factor
            np.clip(arrays["syn_weight"], pc.stdp_w_min, w_max,
                    out=arrays["syn_weight"])


def build_frame_from_arrays(arrays: dict, time_ms: float, step: int) -> SimulationFrame:
    """Build a SimulationFrame from the arrays."""
    voltages = {nid: float(arrays["V"][i]) for i, nid in enumerate(arrays["ids"])}

    active_synapses = []
    if len(arrays["syn_pre"]) > 0:
        g_syn = np.maximum(0, arrays["syn_g_decay"] - arrays["syn_g_rise"])
        V_post = arrays["V"][arrays["syn_post"]]
        currents = g_syn * (V_post - arrays["syn_E_syn"]) * arrays["syn_exc"]
        mask = np.abs(currents) > 0.05
        for idx in np.where(mask)[0]:
            active_synapses.append((
                arrays["ids"][arrays["syn_pre"][idx]],
                arrays["ids"][arrays["syn_post"][idx]],
                float(currents[idx]),
            ))

    active_gaps = []
    if len(arrays["gap_a"]) > 0:
        dV = arrays["V"][arrays["gap_b"]] - arrays["V"][arrays["gap_a"]]
        gap_I = arrays["gap_g"] * dV
        mask = np.abs(gap_I) > 0.05
        for idx in np.where(mask)[0]:
            active_gaps.append((
                arrays["ids"][arrays["gap_a"][idx]],
                arrays["ids"][arrays["gap_b"][idx]],
                float(gap_I[idx]),
            ))

    return SimulationFrame(
        time_ms=time_ms, step=step, voltages=voltages,
        active_synapses=active_synapses, active_gaps=active_gaps,
    )


def get_plasticity_stats(arrays: dict) -> dict:
    """Summary statistics about learning and habituation."""
    if len(arrays["syn_weight"]) == 0:
        return {"weight_mean": 0, "weight_std": 0, "efficacy_mean": 1.0,
                "n_strengthened": 0, "n_weakened": 0, "n_habituated": 0, "n_total": 0}

    w = arrays["syn_weight"]
    w0 = arrays["syn_weight_initial"]
    eff = arrays["syn_efficacy"]

    return {
        "weight_mean": float(w.mean()),
        "weight_std": float(w.std()),
        "weight_change_mean": float((w - w0).mean()),
        "efficacy_mean": float(eff.mean()),
        "efficacy_min": float(eff.min()),
        "n_strengthened": int((w > w0 * 1.05).sum()),
        "n_weakened": int((w < w0 * 0.95).sum()),
        "n_habituated": int((eff < 0.5).sum()),
        "n_total": len(w),
    }
