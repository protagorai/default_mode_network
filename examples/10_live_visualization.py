"""
Example 10: Live C. elegans Neural Visualizer

Three-panel real-time display:
  Left   -- Neural network (anatomical worm layout, glowing neurons)
  Right  -- 2D arena with worm body moving via motor neuron output
  Bottom -- Population activity timeline (sensory / interneuron / motor)

Features:
  - DMN baseline: spontaneous oscillatory activity in interneurons
  - Click a neuron to inject stimulus (80 pA)
  - Click the arena to place a chemical attractant
  - DMN toggle button to see baseline vs stimulus-driven activity
  - Speed control from 0.01x (slow-motion) to 100x (fast)

Usage:
    python examples/10_live_visualization.py
    python examples/10_live_visualization.py --speed 0.1 --duration 60000
    python examples/10_live_visualization.py --no-env --layout grid
    python examples/10_live_visualization.py --backend matplotlib
"""

import argparse

from sdmn.networks.celegans import build_connectome_network
from sdmn.visualization import LiveVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="Live C. elegans neural activation visualizer")
    parser.add_argument("--backend", choices=["browser", "matplotlib"],
                        default="browser")
    parser.add_argument("--layout", choices=["anatomical", "grid", "graph"],
                        default="anatomical")
    parser.add_argument("--speed", type=float, default=0.5,
                        help="Initial speed (default: 0.5 = 2x slow-mo)")
    parser.add_argument("--duration", type=float, default=60000,
                        help="Simulation duration in ms (default: 60000 = 1 min)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-env", action="store_true",
                        help="Disable worm body / arena environment")
    parser.add_argument("--no-dmn", action="store_true",
                        help="Disable default mode network baseline activity")
    args = parser.parse_args()

    print("Loading C. elegans connectome...")
    from sdmn.networks.celegans.network_manager import SimulationConfig, PlasticityConfig
    config = SimulationConfig(
        dt=0.1,
        record_interval=5,
        progress_interval=5000,
        plasticity=PlasticityConfig(
            enable_stdp=True,
            enable_habituation=True,
        ),
    )
    network = build_connectome_network()
    network.config = config

    print(f"Starting visualizer ({args.layout} layout, "
          f"env={'off' if args.no_env else 'on'}, "
          f"DMN={'off' if args.no_dmn else 'on'}, "
          f"speed={args.speed}x, dt={config.dt}ms)...")

    viz = LiveVisualizer(
        network,
        layout=args.layout,
        enable_environment=not args.no_env,
        enable_dmn=not args.no_dmn,
    )

    viz.run(
        duration_ms=args.duration,
        speed=args.speed,
        backend=args.backend,
        port=args.port,
    )


if __name__ == "__main__":
    main()
