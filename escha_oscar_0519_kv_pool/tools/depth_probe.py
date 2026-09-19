"""1本の長チェーンを入れたあと、プレフィクスを段階的に短く切って投げ、
どの深さまで mamba チェックポイントが生きているかを cached_tokens で写像する。"""
import json, random, sys, time, urllib.request
MODEL = open('/tmp/kvbench/model.txt').read().strip()
URL = "http://127.0.0.1:8081/v1/chat/completions"
NFACT = 5200
random.seed(1000)
FACTS = [f"p0fact{i}: {random.randint(0,999)}." for i in range(NFACT)]

def ask(nfact, q="What is p0fact3? Number only."):
    prefix = " ".join(FACTS[:nfact])
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": 8,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role":"system","content":prefix},
                                    {"role":"user","content":q}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d=json.load(r)
    u=d.get("usage",{})
    return time.time()-t0, u.get("prompt_tokens",0), (u.get("prompt_tokens_details") or {}).get("cached_tokens",0)

s,p,c = ask(NFACT)
print(f"seed  full chain: {s:6.1f}s prompt={p} cached={c}", flush=True)
out=[{"nfact":NFACT,"prompt":p,"cached":c,"sec":round(s,1)}]
for frac in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0):
    n = max(1, int(NFACT*frac))
    s,p,c = ask(n)
    print(f"  prefix {frac:5.3f} (nfact={n:5d}): prompt={p:6d} cached={c:6d} hit={100*c/max(p,1):5.1f}% {s:5.1f}s", flush=True)
    out.append({"frac":frac,"nfact":n,"prompt":p,"cached":c,"sec":round(s,1)})
json.dump(out, open(sys.argv[1],"w"), indent=1)
