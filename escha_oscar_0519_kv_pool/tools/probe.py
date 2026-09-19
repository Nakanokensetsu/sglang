import json, sys, time, urllib.request
MODEL = open('/tmp/kvbench/model.txt').read().strip()
URL = "http://127.0.0.1:8081/v1/chat/completions"
PROMPTS = [
    "1から20までの素数を昇順にカンマ区切りで並べてください。説明は不要です。",
    "次を逐語的にそのまま繰り返してください: set LOGFILE=C:\\Users\\Example\\Documents\\app.log",
    "日本の都道府県のうち県庁所在地名が県名と異なるものを5つ挙げてください。",
    "Explain in exactly one sentence what a radix tree is.",
    "12345 * 6789 を計算して数値だけ答えてください。",
]
def ask(p, max_tokens=256):
    body = json.dumps({"model": MODEL, "temperature": 0, "top_p": 1,
                       "seed": 12345, "max_tokens": max_tokens,
                       "messages": [{"role":"user","content":p}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return {"text": d["choices"][0]["message"]["content"],
            "finish": d["choices"][0]["finish_reason"],
            "usage": d.get("usage",{}),
            "sec": round(time.time()-t0,2)}
out=[]
for i,p in enumerate(PROMPTS):
    try:
        out.append({"i":i,"prompt":p,**ask(p)})
        print(f"  [{i}] ok {out[-1]['sec']}s finish={out[-1]['finish']}", flush=True)
    except Exception as e:
        out.append({"i":i,"prompt":p,"error":str(e)}); print(f"  [{i}] ERROR {e}", flush=True)
json.dump(out, open(sys.argv[1],"w"), ensure_ascii=False, indent=1)
print("saved", sys.argv[1])
