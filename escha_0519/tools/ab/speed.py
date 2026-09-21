"""速度計測: 冷prefillスループット / 短文TTFT / decode tok/s / 長文prefill中の短文TTFT。
速度は usage.completion_tokens で測る(ストリームのチャンク数では測らない)。"""
import json, random, sys, time, urllib.request, threading
import os
MODEL = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.txt')).read().strip()
# 2026-09-21: A/B で別ポートのインスタンスを叩けるように
URL = os.environ.get("ESCHA_URL", "http://127.0.0.1:8081/v1/chat/completions")
random.seed(4242)
LONG = " ".join(f"z{i}: {random.randint(0,999)}." for i in range(7000))   # ~90K tok

def call(msgs, max_tokens, stream=False, think=False):
    body = {"model": MODEL, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": think}, "messages": msgs}
    if stream:
        body["stream"] = True; body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        if not stream:
            d = json.load(r); return time.time()-t0, None, d.get("usage", {})
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "): continue
            payload = line[6:]
            if payload == "[DONE]": break
            d = json.loads(payload)
            if ttft is None and d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
                ttft = time.time()-t0
            if d.get("usage"): usage = d["usage"]
    return time.time()-t0, ttft, usage or {}

out = {}
# 1) 冷 prefill スループット
sec, _, u = call([{"role":"system","content":LONG},{"role":"user","content":"ok?"}], 8)
p = u.get("prompt_tokens", 0)
out["cold_prefill"] = {"sec": round(sec,2), "prompt": p, "tok_s": round(p/max(sec,1e-9))}
print(f"1) 冷prefill : {p} tok / {sec:.1f}s = {p/sec:7.1f} tok/s", flush=True)
# 2) 短文 TTFT (bs=1)
ts = []
for _ in range(3):
    _, ttft, _ = call([{"role":"user","content":"Reply with the single word: ok"}], 16, stream=True)
    ts.append(ttft)
out["short_ttft"] = [round(t,3) for t in ts]
print(f"2) 短文TTFT  : {[round(t,3) for t in ts]} s", flush=True)
# 3) decode tok/s (bs=1)
sec, ttft, u = call([{"role":"user","content":"Count from 1 to 400, comma separated."}], 1200, stream=True)
c = u.get("completion_tokens", 0)
out["decode_bs1"] = {"sec": round(sec,2), "completion": c,
                     "tok_s": round(c/max(sec-(ttft or 0),1e-9),1)}
print(f"3) decode bs1: {c} tok / {sec-(ttft or 0):.1f}s = {c/max(sec-(ttft or 0),1e-9):6.1f} tok/s", flush=True)
# 4) 長文prefill中の短文TTFT
res = {}
def bg():
    res["long"] = call([{"role":"system","content":LONG},{"role":"user","content":"ok?"}], 8)
t = threading.Thread(target=bg); t.start(); time.sleep(3)
_, ttft, _ = call([{"role":"user","content":"Reply with the single word: ok"}], 16, stream=True)
out["short_ttft_during_long"] = round(ttft, 3)
print(f"4) 長文中TTFT: {ttft:.3f} s", flush=True)
t.join()
json.dump(out, open(sys.argv[1], "w"), indent=1)
