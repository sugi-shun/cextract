import re
from typing import Any, Dict

import pandas as pd

regex = re.compile(r"\{id([0-9]+)\}")


def resolve(id_str, d, cache):
    if id_str in cache:
        return cache[id_str]

    if id_str not in d:
        return ""

    content = d[id_str]

    def repl(m):
        return resolve(m.group(1), d, cache)

    resolved = regex.sub(repl, content)
    cache[id_str] = resolved
    return resolved


def build(df: pd.DataFrame, threshold: float = 0.8) -> str:
    df_fix = df[df["pred"] > threshold]

    d = {str(idx): c for idx, c in zip(df_fix["id"], df_fix["contents"]) if isinstance(c, str)}

    cache = {}
    final_ids = sorted(int(k) for k in d.keys())
    out = [resolve(str(k), d, cache) for k in final_ids]

    return " ".join(out).replace("[ID_NOT_FOUND]", "")
