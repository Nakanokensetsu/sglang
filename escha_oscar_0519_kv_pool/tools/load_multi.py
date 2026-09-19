"""複数の異なる長チェーンを保持できるかを測る(マルチユーザ想定)。
K 本の別プレフィクスを順に1回ずつ→もう一周。2周目のヒット率が保持率。"""
import json, random, sys, time, urllib.request
MODEL = open('/tmp/kvbench/model.txt').read().strip()
URL = "http://127.0.0.1:8081/v1/chat/completions"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
NFACT = int(sys.argv[3]) if len(sys.argv) > 3 else 5200
prefixes = []
for k in range(K):
    random.seed(1000 + k)
    prefixes.append(" ".join(f"p{k}fact{i}: {random.randint(0,999)}." for i in range(NFACT)))

def ask(k, q):
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": 8,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role":"system","content":prefixes[k]},
                                    {"role":"user","content":q}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d=json.load(r)
    u=d.get("usage",{})
    return time.time()-t0, u.get("prompt_tokens",0), (u.get("prompt_tokens_details") or {}).get("cached_tokens",0)

out=[]
for lap in (1,2,3):
    tot=cac=0; t0=time.time()
    for k in range(K):
        s,p,c = ask(k, f"What is p{k}fact3? Number only.")
        tot+=p; cac+=c
        print(f"  lap{lap} chain{k}: {s:6.1f}s prompt={p} cached={c}", flush=True)
    hit=100*cac/max(tot,1)
    print(f"lap{lap}: K={K} wall={time.time()-t0:6.1f}s hit={hit:5.1f}%", flush=True)
    out.append({"lap":lap,"K":K,"hit":round(hit,1),"prompt":tot,"cached":cac})
json.dump(out, open(sys.argv[1],"w"), indent=1)
