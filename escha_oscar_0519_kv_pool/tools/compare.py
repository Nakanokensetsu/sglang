import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
same = diff = err = 0
for x, y in zip(a, b):
    if "error" in x or "error" in y:
        err += 1; print(f"[{x['i']}] ERROR"); continue
    if x["text"] == y["text"] and x["finish"] == y["finish"]:
        same += 1
    else:
        diff += 1
        print(f"[{x['i']}] DIFF  finish {x['finish']}->{y['finish']}")
        print(f"   A: {x['text'][:160]!r}")
        print(f"   B: {y['text'][:160]!r}")
print(f"identical={same} differ={diff} errors={err}")
sys.exit(1 if (diff or err) else 0)
