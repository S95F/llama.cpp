#!/usr/bin/env python3
"""EXP-0007: EXP-0006's summarize.py on this experiment's files.

    python summarize.py <tag>... --cap C:/nsys/exp0007
    python summarize.py --table --cap C:/nsys/exp0007
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("summarize6", os.path.join(HERE, "..", "EXP-0006", "summarize.py"))
s6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s6)
s6.HERE = HERE  # read and write this experiment's files

if __name__ == "__main__":
    if not any(x == "--cap" or x.startswith("--cap=") for x in sys.argv[1:]):
        sys.argv += ["--cap", "C:/nsys/exp0007"]
    sys.exit(s6.main())
