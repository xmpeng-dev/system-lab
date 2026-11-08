"""python3 parse_e2e.py: per-step ms (median of iterations 4-12) and loss per E2E log."""
import re
import statistics
import sys
from pathlib import Path

E = Path("/home/xiaompen/uly_reg/e2e")
IT = re.compile(r"iteration\s+(\d+)/.*?elapsed time per iteration \(ms\): ([\d.]+).*?lm loss: ([\d.E+-]+)")


def parse(f):
    t = f.read_text(errors="replace")
    its = {int(m[1]): (float(m[2]), float(m[3])) for m in IT.finditer(t)}
    steady = [its[i][0] for i in range(4, 13) if i in its]
    head = re.search(r"\[pyboot\] sys.path head: \[[^\]]*'([^']*turbo_[^']*)'", t)
    return dict(
        ms=statistics.median(steady) if steady else float("nan"),
        sd=statistics.pstdev(steady) if len(steady) > 1 else 0,
        loss=its.get(12, (0, float("nan")))[1],
        n=len(its),
        applied="Applied: megatron.ulysses_fused_a2a" in t,
        stock="keeping the stock path" in t,
        turbo=Path(head[1]).name if head else "?",
        ok="exit=0" in t,
    )


if __name__ == "__main__":
    pat = sys.argv[1] if len(sys.argv) > 1 else "*"
    for f in sorted(E.glob(f"{pat}.log")):
        r = parse(f)
        print(f"{f.stem:<32} {r['ms']:8.1f} ±{r['sd']:5.1f}  loss12={r['loss']:.6f}  its={r['n']:2d}  applied={int(r['applied'])} stock={int(r['stock'])} {r['turbo']} ok={int(r['ok'])}")
