"""Quick WebSocket performance check."""
import asyncio, json, time, websockets

async def test():
    ws = await websockets.connect("ws://127.0.0.1:8080/ws")
    init = json.loads(await ws.recv())
    print(f"Init OK: {len(init['neurons'])} neurons, {len(init['edges'])} edges")

    t0 = time.time()
    frames = []
    for _ in range(10):
        f = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        frames.append(f)
    elapsed = time.time() - t0
    print(f"Got {len(frames)} frames in {elapsed:.2f}s = {len(frames)/elapsed:.1f} fps")

    f = frames[-1]
    step, tm = f["step"], f["time_ms"]
    print(f"Last frame: step={step}  time={tm:.1f} ms")

    bri = [(v + 65) / 45 for v in f["voltages"].values()]
    print(f"Brightness: min={min(bri):.3f}  max={max(bri):.3f}  mean={sum(bri)/len(bri):.3f}")
    print(f"Active synapses: {len(f.get('synapses', []))}")
    print(f"Active gaps: {len(f.get('gaps', []))}")

    env = f.get("env")
    if env:
        print(f"Worm speed: {env['speed']:.1f}  chem: {env['chem']:.3f}")

    tl = f.get("timeline")
    if tl:
        print(f"Timeline points: {len(tl['times'])}")

    # Measure sim advance rate
    t1_step, t1_time = frames[0]["step"], frames[0]["time_ms"]
    sim_advance = tm - t1_time
    print(f"Sim advanced {sim_advance:.1f} ms in {elapsed:.2f}s wall = {sim_advance/elapsed/1000:.2f}x realtime")

    await ws.close()
    print("ALL OK")

asyncio.run(test())
