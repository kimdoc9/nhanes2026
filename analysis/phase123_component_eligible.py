import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import phase123 as p


def numeric_fixed(s: pd.Series) -> pd.Series:
    """Normalize the pandas/SAS-XPORT tiny sentinel used when numeric zero is decoded."""
    x = pd.to_numeric(s, errors="coerce")
    return x.mask(x.notna() & (x.abs() < 1e-50), 0.0)


p.numeric = numeric_fixed

# Component eligibility is structural, not item-level missingness. DPQ documentation
# excludes proxy-required participants; absence from the DPQ component file must not
# be counted as item nonresponse. Apply this correction before any network fitting.
AMENDMENT = {
    "date": "2026-08-20",
    "type": "source-eligibility operationalization correction before Phase 3 network fitting",
    "rationale": (
        "NHANES DPQ is administered at the MEC and participants requiring a proxy are not eligible. "
        "Therefore the missingness denominator is adults aged >=65 with a record in both DPQ and SLQ "
        "component files, rather than all age-eligible DEMO records. The prespecified 15% node-missingness "
        "and 70% complete-case-retention gates are unchanged."
    ),
    "eligibility": "RIDAGEYR>=65 AND SEQN present in DPQ component AND SEQN present in SLQ component",
    "gate_change": "none",
    "network_or_outcome_results_seen_before_correction": False,
    "xport_zero_correction": "numeric values with absolute magnitude <1e-50 are normalized to 0 before recoding",
}
canon = json.dumps(AMENDMENT, sort_keys=True, separators=(",", ":")).encode()
AMENDMENT["amendment_sha256"] = hashlib.sha256(canon).hexdigest()
(p.OUT / "protocol_amendment_component_eligibility.json").write_text(json.dumps(AMENDMENT, indent=2), encoding="utf-8")


def build_cycle_component_eligible(cycle: str, cfg: dict, manifest: list[dict]):
    suffix = cfg["suffix"]
    frames = {}
    for comp in p.COMPONENTS:
        url = f"{cfg['base']}/{comp}_{suffix}.XPT"
        dest = p.RAW / f"{comp}_{suffix}.XPT"
        content, meta = p.download_xpt(url, dest)
        df = p.read_xpt_bytes(content)
        meta.update({"cycle": cycle, "component": comp, "rows": int(len(df)), "columns": int(df.shape[1])})
        manifest.append(meta)
        frames[comp] = df

    demo = frames["DEMO"].copy()
    dpq = frames["DPQ"].copy()
    slq = frames["SLQ"].copy()
    for df in (demo, dpq, slq):
        df["SEQN"] = numeric_fixed(df["SEQN"]).astype("Int64")

    dpq_seqn = set(dpq["SEQN"].dropna().astype(int).tolist())
    slq_seqn = set(slq["SEQN"].dropna().astype(int).tolist())

    keep_demo = [c for c in ["SEQN", "RIDAGEYR", "RIAGENDR", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"] if c in demo.columns]
    keep_dpq = ["SEQN"] + sorted({v for _, v, _, _ in p.NODE_SPECS if v.startswith("DPQ")})
    keep_slq = ["SEQN"] + sorted({v for _, v, _, _ in p.NODE_SPECS if v.startswith("SL")})

    merged = demo[keep_demo].merge(dpq[keep_dpq], on="SEQN", how="left", validate="one_to_one")
    merged = merged.merge(slq[keep_slq], on="SEQN", how="left", validate="one_to_one")
    merged["RIDAGEYR"] = numeric_fixed(merged["RIDAGEYR"])

    age65 = merged.loc[merged["RIDAGEYR"] >= 65].copy()
    age65["_DPQ_RECORD"] = age65["SEQN"].astype(int).isin(dpq_seqn)
    age65["_SLQ_RECORD"] = age65["SEQN"].astype(int).isin(slq_seqn)
    component_eligible = age65["_DPQ_RECORD"] & age65["_SLQ_RECORD"]
    eligible = age65.loc[component_eligible].copy()
    eligible["cycle"] = cycle

    structural = {
        "cycle": cycle,
        "age65_demo_n": int(len(age65)),
        "age65_with_dpq_record_n": int(age65["_DPQ_RECORD"].sum()),
        "age65_with_slq_record_n": int(age65["_SLQ_RECORD"].sum()),
        "analysis_component_eligible_n": int(component_eligible.sum()),
        "excluded_no_dpq_record_n": int((~age65["_DPQ_RECORD"]).sum()),
        "excluded_no_slq_record_n": int((~age65["_SLQ_RECORD"]).sum()),
        "excluded_not_in_both_components_n": int((~component_eligible).sum()),
    }
    (p.OUT / f"structural_eligibility_{cycle.replace('-', '_')}.json").write_text(json.dumps(structural, indent=2), encoding="utf-8")

    for name, raw_var, domain, rule in p.NODE_SPECS:
        eligible[name] = p.recode_node(eligible[raw_var], rule)

    keep_ready = ["SEQN", "cycle", "RIDAGEYR", "RIAGENDR", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"] + p.NODES
    ready_all = eligible[keep_ready].copy()
    complete = ready_all.dropna(subset=p.NODES).copy()
    for n in p.NODES:
        complete[n] = complete[n].astype(int)

    audit_rows = []
    weights = numeric_fixed(ready_all["WTMEC2YR"])
    cc_weights = numeric_fixed(complete["WTMEC2YR"])
    for n in p.NODES:
        obs = ready_all[n].notna()
        prev_u = float(ready_all.loc[obs, n].mean()) if obs.any() else np.nan
        w = weights.loc[obs]
        y = ready_all.loc[obs, n]
        prev_w = float(np.average(y, weights=w)) if obs.any() and np.isfinite(w).all() and w.sum() > 0 else np.nan
        cc_prev_u = float(complete[n].mean()) if len(complete) else np.nan
        cc_prev_w = float(np.average(complete[n], weights=cc_weights)) if len(complete) and np.isfinite(cc_weights).all() and cc_weights.sum() > 0 else np.nan
        audit_rows.append({
            "cycle": cycle,
            "node": n,
            "domain": p.DOMAINS[n],
            "eligible_n": int(len(ready_all)),
            "observed_n": int(obs.sum()),
            "missing_n": int((~obs).sum()),
            "missing_fraction": float((~obs).mean()),
            "prevalence_observed_unweighted": prev_u,
            "prevalence_observed_weighted": prev_w,
            "complete_case_n": int(len(complete)),
            "prevalence_complete_case_unweighted": cc_prev_u,
            "prevalence_complete_case_weighted": cc_prev_w,
            "extreme_prevalence_flag": bool((prev_u < 0.02) or (prev_u > 0.98)) if np.isfinite(prev_u) else True,
        })
    return ready_all, complete, pd.DataFrame(audit_rows)


p.build_cycle = build_cycle_component_eligible
p.main()
