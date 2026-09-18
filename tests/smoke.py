"""End-to-end check against the bundled samples: python tests/smoke.py"""

import json
from pathlib import Path

from indic_asr import Transcriber

refs = json.loads(Path("samples/refs.json").read_text())
files = list(refs)
t = Transcriber()
for dec in ("ctc", "rnnt"):
    print(f"== {dec}")
    for f, hyp in zip(files, t.transcribe(files, decoder=dec)):
        mark = "OK " if hyp == refs[f] else "~~ "
        print(f"{mark}{f}\n   hyp: {hyp}\n   ref: {refs[f]}")
