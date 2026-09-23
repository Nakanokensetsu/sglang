"""並列10で VRAM 崖を探す。メモの判別法: 全力=97%/204W/73℃、崖=100%/45W/48℃。"""

import concurrent.futures as cf
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request

MODEL = (
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.txt"))
    .read()
    .strip()
)
# 2026-09-21: A/B で別ポートのインスタンスを叩けるように
URL = os.environ.get("ESCHA_URL", "http://127.0.0.1:8081/v1/chat/completions")
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 10
random.seed(77)
# 各リクエストに別プレフィクス(共有させない=いちばん重い条件)
PREFIXES = [
    " ".join(f"c{k}w{i}: {random.randint(0,999)}." for i in range(1500))
    for k in range(CONC)
]
stop = False
samples = []


def sampler():
    while not stop:
        try:
            o = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,utilization.gpu,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            samples.append(o.replace("\n", " | "))
        except Exception:
            pass
        time.sleep(3)


th = threading.Thread(target=sampler)
th.start()


def ask(k):
    body = json.dumps(
        {
            "model": MODEL,
            "temperature": 0,
            "max_tokens": 200,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": PREFIXES[k]},
                {"role": "user", "content": f"Reply with the number {k}."},
            ],
        }
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1200) as r:
            d = json.load(r)
        u = d.get("usage", {})
        return {
            "k": k,
            "sec": round(time.time() - t0, 2),
            "prompt": u.get("prompt_tokens", 0),
            "completion": u.get("completion_tokens", 0),
        }
    except Exception as e:
        return {"k": k, "sec": round(time.time() - t0, 2), "error": str(e)[:120]}


t0 = time.time()
with cf.ThreadPoolExecutor(CONC) as ex:
    res = list(ex.map(ask, range(CONC)))
# 2周目(キャッシュ温)
with cf.ThreadPoolExecutor(CONC) as ex:
    res2 = list(ex.map(ask, range(CONC)))
stop = True
th.join()
err = [r for r in res + res2 if "error" in r]
comp = sum(r.get("completion", 0) for r in res + res2)
print(f"並列{CONC}: wall={time.time()-t0:.1f}s errors={len(err)} completion_tok={comp}")
print(
    f"  1周目 最遅={max(r['sec'] for r in res):.1f}s 最速={min(r['sec'] for r in res):.1f}s"
)
print(
    f"  2周目 最遅={max(r['sec'] for r in res2):.1f}s 最速={min(r['sec'] for r in res2):.1f}s"
)
for e in err[:3]:
    print("  ERR:", e)
print("  VRAM/util/power/temp サンプル(末尾5):")
for s in samples[-5:]:
    print("   ", s)
json.dump(
    {"res": res, "res2": res2, "samples": samples}, open(sys.argv[1], "w"), indent=1
)
