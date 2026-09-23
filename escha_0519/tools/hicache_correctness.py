"""HiCache のホスト層復元が**内容を壊していないか**を判定する(2026-09-20 追加)。

第24節で「ホスト層から復元した prefix が別パスの声で間違った出力を返す。
クラッシュも警告も会計異常も無く cached_tokens は満額」= 静かに壊れる型、と記録した。
その型を機械判定するためのテスト。

やり方:
  * K 本の長チェーンを作る。各 fact の値は seed から**こちらが知っている**ので、
    自己一致(前回と同じ答え)ではなく**正解との一致**で見られる。
  * フェーズ1: 各チェーンを1回ずつ。デバイス側だけで完結する状態の正答率。
    ここが低いとテスト自体に判別力が無いので、その時点で中止する。
  * フェーズ2: 対象チェーンを1本置いて、残りのチェーンを回してデバイスプールから
    追い出す(= ホストへ退避させる)。そのうえで対象チェーンに戻り、
    **ホストから復元された prefix** で同じ質問をする。正解と一致するか、
    かつ cached_tokens がヒットを報告しているかを見る。
  * これを ROUNDS 回。

使い方:
  python3 hicache_correctness.py out.json [PORT] [ROUNDS] [K] [NFACT]

デバイスプールを小さくして起動していないと、フェーズ2で追い出しが起きず
「ホストを経由していないのに合格」になる。判定のために
cached-from-host が実際に起きたかは、サーバ側のログ
(`hicache` / `backup` 行)と合わせて見ること。
"""

import json
import os
import random
import sys
import time
import urllib.request

MODEL = (
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.txt"))
    .read()
    .strip()
)
OUT = sys.argv[1]
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8081
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
K = int(sys.argv[4]) if len(sys.argv) > 4 else 3
NFACT = int(sys.argv[5]) if len(sys.argv) > 5 else 2600
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

prefixes, truth = [], []
for k in range(K):
    rnd = random.Random(1000 + k)
    vals = [rnd.randint(0, 999) for _ in range(NFACT)]
    prefixes.append(" ".join(f"p{k}fact{i}: {v}." for i, v in enumerate(vals)))
    # 途中の fact を聞く。先頭/末尾だと位置バイアスで当たってしまう。
    truth.append((NFACT // 2, vals[NFACT // 2]))


def ask(k):
    idx, _ = truth[k]
    body = json.dumps(
        {
            "model": MODEL,
            "temperature": 0,
            "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": prefixes[k]},
                {
                    "role": "user",
                    "content": f"What is p{k}fact{idx}? Answer with the number only.",
                },
            ],
        }
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    u = d.get("usage", {})
    text = d["choices"][0]["message"]["content"].strip()
    return {
        "text": text,
        "secs": round(time.time() - t0, 2),
        "prompt": u.get("prompt_tokens", 0),
        "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
    }


def correct(k, res):
    want = str(truth[k][1])
    got = "".join(c for c in res["text"] if c.isdigit())
    return got == want


def main():
    print(f"model={MODEL} port={PORT} K={K} NFACT={NFACT} rounds={ROUNDS}")
    print(f"prefix ~{len(prefixes[0].split())} words/chain\n")

    # --- フェーズ1: 判別力の確認 ---
    base = []
    for k in range(K):
        r = ask(k)
        ok = correct(k, r)
        base.append(ok)
        print(
            f"  phase1 chain{k}: {'OK ' if ok else 'NG '} got={r['text']!r} "
            f"want={truth[k][1]} prompt={r['prompt']} cached={r['cached']} {r['secs']}s",
            flush=True,
        )
    if not all(base):
        print(
            "\n中止: デバイス側だけでも正解しない。このテストには判別力が無い。"
            "\nNFACT を下げるか、質問位置を変えること。"
        )
        json.dump(
            {"aborted": "phase1 failed", "phase1": base}, open(OUT, "w"), indent=1
        )
        return 2

    # --- フェーズ2: ホストからの復元 ---
    rows, bad = [], 0
    for rnd_i in range(ROUNDS):
        target = rnd_i % K
        # 対象以外を回してデバイスプールから追い出す
        for k in range(K):
            if k != target:
                ask(k)
        r = ask(target)
        ok = correct(target, r)
        if not ok:
            bad += 1
        rows.append(
            {
                "round": rnd_i,
                "chain": target,
                "ok": ok,
                "got": r["text"],
                "want": truth[target][1],
                "prompt": r["prompt"],
                "cached": r["cached"],
                "secs": r["secs"],
            }
        )
        print(
            f"  round{rnd_i:3d} chain{target}: {'OK ' if ok else 'NG '} "
            f"got={r['text']!r} want={truth[target][1]} "
            f"cached={r['cached']}/{r['prompt']} {r['secs']}s",
            flush=True,
        )

    hit = sum(x["cached"] for x in rows) / max(sum(x["prompt"] for x in rows), 1)
    print(f"\n正答 {ROUNDS - bad}/{ROUNDS}  平均キャッシュヒット {100*hit:.1f}%")
    if bad:
        print("=> 静かに壊れている。ホスト層の復元が内容を変えた。")
    elif hit < 0.5:
        print(
            "=> 全問正解だが、ヒット率が低い。追い出しが起きておらず"
            "ホスト経路を通っていない疑い。プールをもっと小さくして再測すること。"
        )
    else:
        print("=> ホストから復元した prefix で内容が保たれている。")
    json.dump(
        {"phase1": base, "rounds": rows, "wrong": bad, "hit": round(hit, 4)},
        open(OUT, "w"),
        indent=1,
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
