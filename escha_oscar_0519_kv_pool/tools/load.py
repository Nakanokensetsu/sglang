"""負荷プローブ: (a) 共有長プレフィクス + 分岐、(b) 短い完了リクエスト多数。
usage.cached_tokens で prefix ヒットを測る(--enable-cache-report 前提)。"""
import json, random, sys, time, urllib.request, concurrent.futures as cf
MODEL = open('/tmp/kvbench/model.txt').read().strip()
URL = "http://127.0.0.1:8081/v1/chat/completions"
random.seed(0)
PREFIX = " ".join(f"fact{i}: the value is {random.randint(0,999)}." for i in range(1200))

def ask(msgs, max_tokens=16):
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": max_tokens,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": msgs}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type":"application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    u = d.get("usage", {})
    return {"sec": round(time.time()-t0, 2),
            "prompt": u.get("prompt_tokens", 0),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)}

def phase(name, jobs, conc):
    t0 = time.time()
    with cf.ThreadPoolExecutor(conc) as ex:
        res = list(ex.map(lambda m: ask(m), jobs))
    tot = sum(r["prompt"] for r in res); cac = sum(r["cached"] for r in res)
    print(f"{name:26s} n={len(res):3d} conc={conc:2d} wall={time.time()-t0:6.1f}s "
          f"prompt_tok={tot:7d} cached={cac:7d} hit={100*cac/max(tot,1):5.1f}%", flush=True)
    return {"name": name, "n": len(res), "conc": conc, "hit": 100*cac/max(tot,1),
            "wall": round(time.time()-t0,1), "prompt": tot, "cached": cac}

out = []
# 1) 同一プレフィクスの繰り返し(ヒットするはず)
same = [[{"role":"system","content":PREFIX},{"role":"user","content":"What is fact7? Number only."}] for _ in range(12)]
out.append(phase("A: identical prefix", same, 1))
out.append(phase("A: identical prefix x8", same, 8))
# 2) 分岐: 同じプレフィクス + 異なる中間文 + 異なる質問(#36935 の対象)
forks = []
for i in range(12):
    mid = " ".join(f"note{i}_{j}: {random.randint(0,999)}." for j in range(200))
    forks.append([{"role":"system","content":PREFIX},
                  {"role":"user","content":mid + f"\nWhat is fact{i}? Number only."}])
out.append(phase("B: forked prefix", forks, 1))
out.append(phase("B: forked prefix x8", forks, 8))
# 3) 短い完了リクエスト多数(長さ0 insert 経路)
short = [[{"role":"user","content":f"Say the number {i}."}] for i in range(40)]
out.append(phase("C: short finished", short, 8))
json.dump(out, open(sys.argv[1],"w"), indent=1)
