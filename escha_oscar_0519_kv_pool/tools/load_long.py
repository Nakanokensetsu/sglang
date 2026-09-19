"""#36935 再現用: 長いプレフィクス(既定 ~60K tok)を共有して分岐させる。
チェックポイント需要 ≈ prefix_len / chunked_prefill_size(4096)。"""
import json, random, sys, time, urllib.request, concurrent.futures as cf
MODEL = open('/tmp/kvbench/model.txt').read().strip()
URL = "http://127.0.0.1:8081/v1/chat/completions"
NFACT = int(sys.argv[2]) if len(sys.argv) > 2 else 5200   # ~40K tok(1 fact ≈ 7.6 tok)
NREQ  = int(sys.argv[3]) if len(sys.argv) > 3 else 6
random.seed(0)
PREFIX = " ".join(f"fact{i}: the value is {random.randint(0,999)}." for i in range(NFACT))

def ask(msgs, max_tokens=8):
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": max_tokens,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": msgs}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    u = d.get("usage", {})
    return {"sec": round(time.time()-t0,2), "prompt": u.get("prompt_tokens",0),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens",0)}

def phase(name, jobs, conc):
    t0=time.time()
    with cf.ThreadPoolExecutor(conc) as ex:
        res=list(ex.map(lambda m: ask(m), jobs))
    tot=sum(r["prompt"] for r in res); cac=sum(r["cached"] for r in res)
    print(f"{name:30s} n={len(res)} conc={conc} wall={time.time()-t0:7.1f}s "
          f"prompt={tot:8d} cached={cac:8d} hit={100*cac/max(tot,1):5.1f}%", flush=True)
    return {"name":name,"conc":conc,"hit":round(100*cac/max(tot,1),1),
            "wall":round(time.time()-t0,1),"prompt":tot,"cached":cac}

out=[]
warm=[[{"role":"system","content":PREFIX},{"role":"user","content":"What is fact3? Number only."}]]
w=phase("warmup (cold prefill)", warm, 1)
out.append(w)
assert w["prompt"] < 110000, f"prompt {w['prompt']} が文脈上限に当たって切り詰められている"
same=[[{"role":"system","content":PREFIX},{"role":"user","content":f"What is fact{i}? Number only."}] for i in range(NREQ)]
out.append(phase("A: identical prefix", same, 1))
out.append(phase("A: identical prefix x6", same, NREQ))
forks=[]
for i in range(NREQ):
    mid=" ".join(f"note{i}_{j}: {random.randint(0,999)}." for j in range(300))
    forks.append([{"role":"system","content":PREFIX},
                  {"role":"user","content":mid+f"\nWhat is fact{i}? Number only."}])
out.append(phase("B: forked prefix", forks, 1))
out.append(phase("B: forked prefix x6", forks, NREQ))
out.append(phase("A again (after forks)", same, 1))
json.dump(out, open(sys.argv[1],"w"), indent=1)
