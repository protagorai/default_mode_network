"""
Browser-based live visualisation backend with environment and stimuli.

Starlette + WebSocket server that pushes neural, environment, and timeline
frames to the browser and receives speed / pause / stimulus commands.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sdmn.visualization.live_visualizer import LiveVisualizer

_STATIC_DIR = Path(__file__).resolve().parent / "static"


class BrowserVisualizer:

    def __init__(self, controller: "LiveVisualizer",
                 host: str = "127.0.0.1", port: int = 8080):
        self.ctrl = controller
        self.host = host
        self.port = port

    def serve(self) -> None:
        try:
            import uvicorn
        except ImportError:
            print("Browser backend requires uvicorn.")
            print("Install with:  poetry install --with web")
            return

        from starlette.applications import Starlette
        from starlette.responses import HTMLResponse
        from starlette.routing import Route, WebSocketRoute
        from starlette.websockets import WebSocket, WebSocketDisconnect

        ctrl = self.ctrl

        chem_field = None
        if ctrl.environment:
            chem_field = ctrl.environment.get_chemical_field(resolution=25)

        init_payload = json.dumps({
            "type": "init",
            "neurons": ctrl.get_positions_list(),
            "edges": ctrl.get_edges(),
            "regions": ctrl.get_region_labels(),
            "hasEnv": ctrl.environment is not None,
            "chemField": chem_field,
            "arena": [ctrl.environment.arena_w, ctrl.environment.arena_h] if ctrl.environment else None,
        })

        async def index(request):
            html_path = _STATIC_DIR / "index.html"
            return HTMLResponse(html_path.read_text(encoding="utf-8"))

        async def ws_endpoint(ws: WebSocket):
            await ws.accept()
            await ws.send_text(init_payload)

            async def _sender():
                last_step = -1
                while not ctrl._stop.is_set():
                    vf = ctrl.get_latest_frame()
                    if vf and vf.sim.step != last_step:
                        last_step = vf.sim.step
                        msg = {
                            "type": "frame",
                            "time_ms": round(vf.sim.time_ms, 3),
                            "step": vf.sim.step,
                            "voltages": {k: round(v, 2) for k, v in vf.sim.voltages.items()},
                            "synapses": [[a, b, round(c, 2)] for a, b, c in vf.sim.active_synapses],
                            "gaps": [[a, b, round(c, 2)] for a, b, c in vf.sim.active_gaps],
                            "paused": ctrl.is_paused,
                            "speed": round(ctrl.speed, 4),
                            "dmn": vf.dmn_active,
                            "engine": ctrl.engine,
                            "dmn_mode": ctrl.dmn_mode,
                        }
                        if vf.env:
                            msg["env"] = vf.env
                        if vf.timeline:
                            msg["timeline"] = vf.timeline
                        if vf.plasticity:
                            msg["plasticity"] = vf.plasticity
                        if vf.neuromod:
                            msg["neuromod"] = vf.neuromod
                        await ws.send_text(json.dumps(msg))
                    await asyncio.sleep(1 / 60)

            sender_task = asyncio.ensure_future(_sender())

            try:
                while True:
                    data = await ws.receive_text()
                    cmd = json.loads(data)
                    action = cmd.get("action")
                    if action == "set_speed":
                        ctrl.set_speed(float(cmd["value"]))
                    elif action == "pause":
                        ctrl.pause()
                    elif action == "resume":
                        ctrl.resume()
                    elif action == "step":
                        ctrl.step_once()
                    elif action == "stop":
                        ctrl.stop()
                        break
                    elif action == "stimulus":
                        nid = cmd.get("neuron")
                        cur = float(cmd.get("current", 80.0))
                        if nid:
                            ctrl.inject_stimulus(nid, cur)
                    elif action == "add_chemical":
                        x = float(cmd.get("x", 250))
                        y = float(cmd.get("y", 200))
                        ctrl.add_chemical_source(x, y, strength=1.0, radius=80.0)
                    elif action == "add_food":
                        x = float(cmd.get("x", 250))
                        y = float(cmd.get("y", 200))
                        if ctrl.environment:
                            ctrl.environment.add_food_source(x, y)
                    elif action == "add_pain":
                        x = float(cmd.get("x", 250))
                        y = float(cmd.get("y", 200))
                        if ctrl.environment:
                            ctrl.environment.add_pain_zone(x, y)
                    elif action == "toggle_neuromod":
                        feature = cmd.get("feature")
                        if ctrl.neuromod and feature:
                            cur = getattr(ctrl.neuromod.config, feature, None)
                            if isinstance(cur, bool):
                                setattr(ctrl.neuromod.config, feature, not cur)
                    elif action == "toggle_dmn":
                        ctrl.enable_dmn = not ctrl.enable_dmn
                    elif action == "set_engine":
                        ctrl.set_engine(cmd.get("value", "vectorized"))
                    elif action == "set_dmn_mode":
                        ctrl.set_dmn_mode(cmd.get("value", "oscillator"))
                    elif action == "restart":
                        ctrl.restart()
                    elif action == "clear_chemicals":
                        if ctrl.environment:
                            ctrl.environment.clear_chemical_sources()
            except WebSocketDisconnect:
                pass
            finally:
                sender_task.cancel()

        app = Starlette(routes=[
            Route("/", index),
            WebSocketRoute("/ws", ws_endpoint),
        ])

        def _open_browser():
            time.sleep(1.5)
            webbrowser.open(f"http://{self.host}:{self.port}")

        threading.Thread(target=_open_browser, daemon=True).start()

        uvicorn.run(app, host=self.host, port=self.port,
                    log_level="warning")
