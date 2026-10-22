#!/usr/bin/env python3
"""Execute notebooks/end_to_end_demo.ipynb and save stdout into cell outputs."""

from __future__ import annotations

import json
import sys
import traceback
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
NB = ROOT / "notebooks" / "end_to_end_demo.ipynb"


def main() -> int:
    nb = json.loads(NB.read_text())
    g: dict = {"__name__": "__main__"}
    n = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        n += 1
        src = "".join(cell.get("source") or [])
        buf = StringIO()
        old = sys.stdout
        sys.stdout = buf
        err = None
        try:
            exec(compile(src, f"<cell{n}>", "exec"), g, g)
        except Exception:
            err = traceback.format_exc()
            print(err)
        finally:
            sys.stdout = old
        text = buf.getvalue()
        outputs = []
        if text:
            outputs.append(
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": text.splitlines(keepends=True),
                }
            )
        if err:
            outputs.append(
                {
                    "output_type": "error",
                    "ename": "Exception",
                    "evalue": err.splitlines()[-1],
                    "traceback": err.splitlines(),
                }
            )
        cell["outputs"] = outputs
        cell["execution_count"] = n
    NB.write_text(json.dumps(nb, indent=1) + "\n")
    print("executed →", NB)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
