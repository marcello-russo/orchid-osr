"""Mapping from local experiment directory names to HuggingFace repo paths.

Training uses historical prefixes (exp1..exp5), while the HF models repo
stores the same runs under different directory prefixes (exp6..exp9).
Keeping the mapping in one place avoids building wrong remote paths in
weights, adapters and OSR stats.
"""

from __future__ import annotations

# local training prefix -> directory prefix in the HF models repo
HF_PREFIX_MAP: dict[str, str] = {
    "exp1_ce": "exp6_ce",
    "exp2_supcon": "exp7_supcon",
    "exp3_sphor": "exp8_sphor",
    "exp4_sphor_ncd": "exp9_ncd",
    "exp5_sphor_ncd": "exp5_sphor_ncd",
}


def local_prefix(dir_name: str) -> str | None:
    for prefix in HF_PREFIX_MAP:
        if dir_name.startswith(prefix):
            return prefix
    return None


def hf_prefix(prefix: str) -> str:
    return HF_PREFIX_MAP.get(prefix, prefix)


def hf_exp_dir(dir_name: str) -> str | None:
    """``exp2_supcon_convnext_small`` -> ``exp7_supcon/exp7_supcon_convnext_small``."""
    prefix = local_prefix(dir_name)
    if prefix is None:
        return None
    remote = hf_prefix(prefix)
    return f"{remote}/{remote}{dir_name[len(prefix):]}"


def hf_file(dir_name: str, filename: str) -> str | None:
    base = hf_exp_dir(dir_name)
    return f"{base}/{filename}" if base else None
