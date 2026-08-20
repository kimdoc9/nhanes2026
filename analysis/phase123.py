from __future__ import annotations

import hashlib
import json
import math
import time
from io import BytesIO
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.special import expit, gammaln, logsumexp
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

TITLE = "Credibility-Aware Virtual Perturbation of Late-Life Depression–Sleep Symptom Systems Using an Ising Model"
RUN_DATE = "2026-08-20"
OUT = Path("artifact")
RAW = Path("raw_tmp")
OUT.mkdir(exist_ok=True)
RAW.mkdir(exist_ok=True)

CYCLES = {
    "2015-2016": {"suffix": "I", "base": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles"},
    "2017-2018": {"suffix": "J", "base": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"},
}
COMPONENTS = ["DEMO", "DPQ", "SLQ"]

NODE_SPECS = [
    ("D1_anhedonia", "DPQ010", "depression", "phq_ge1"),
    ("D2_depressed_mood", "DPQ020", "depression", "phq_ge1"),
    ("D3_fatigue", "DPQ040", "depression", "phq_ge1"),
    ("D4_appetite", "DPQ050", "depression", "phq_ge1"),
    ("D5_worthlessness", "DPQ060", "depression", "phq_ge1"),
    ("D6_concentration", "DPQ070", "depression", "phq_ge1"),
    ("D7_psychomotor", "DPQ080", "depression", "phq_ge1"),
    ("D8_death_self_harm", "DPQ090", "depression", "phq_ge1"),
    ("S1_sleep_disturbance", "DPQ030", "sleep", "phq_ge1"),
    ("S2_short_sleep", "SLD012", "sleep", "sleep_lt7"),
    ("S3_frequent_snoring", "SLQ030", "sleep", "slq_freq_ge2"),
    ("S4_sdb_symptom", "SLQ040", "sleep", "slq_freq_ge2"),
    ("S5_daytime_sleepiness", "SLQ120", "sleep", "slq_sleepy_ge3"),
]
NODES = [x[0] for x in NODE_SPECS]
DOMAINS = {x[0]: x[2] for x in NODE_SPECS}
P = len(NODES)
GAMMA = 0.5
C_GRID = np.logspace(-2.5, 1.5, 25)
BOOT_B = 500
BOOT_SEED = 20260820
ZERO_TOL = 1e-8


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_xpt(url: str, dest: Path, attempts: int = 4) -> tuple[bytes, dict]:
    last = None
    headers = {"User-Agent": "Mozilla/5.0 NHANES-reproducible-research/1.0"}
    for i in range(attempts):
        try:
            r = requests.get(url, timeout=90, headers=headers)
            r.raise_for_status()
            content = r.content
            if len(content) < 1000:
                raise RuntimeError(f"Unexpectedly small response ({len(content)} bytes)")
            dest.write_bytes(content)
            return content, {
                "url": url,
                "status_code": r.status_code,
                "bytes": len(content),
                "sha256": sha256_bytes(content),
                "content_type": r.headers.get("content-type"),
            }
        except Exception as e:
            last = e
            time.sleep(2 ** i)
    raise RuntimeError(f"Failed to download {url}: {last}")


def read_xpt_bytes(content: bytes) -> pd.DataFrame:
    return pd.read_sas(BytesIO(content), format="xport", encoding="utf-8")


def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def recode_node(s: pd.Series, rule: str) -> pd.Series:
    x = numeric(s)
    out = pd.Series(np.nan, index=x.index, dtype="float64")
    if rule == "phq_ge1":
        ok = x.isin([0, 1, 2, 3])
        out.loc[ok] = (x.loc[ok] >= 1).astype(float)
    elif rule == "sleep_lt7":
        ok = x.notna() & (x >= 0) & (x <= 24)
        out.loc[ok] = (x.loc[ok] < 7).astype(float)
    elif rule == "slq_freq_ge2":
        ok = x.isin([0, 1, 2, 3])
        out.loc[ok] = (x.loc[ok] >= 2).astype(float)
    elif rule == "slq_sleepy_ge3":
        ok = x.isin([0, 1, 2, 3, 4])
        out.loc[ok] = (x.loc[ok] >= 3).astype(float)
    else:
        raise ValueError(rule)
    return out


def build_cycle(cycle: str, cfg: dict, manifest: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    suffix = cfg["suffix"]
    frames = {}
    for comp in COMPONENTS:
        url = f"{cfg['base']}/{comp}_{suffix}.XPT"
        dest = RAW / f"{comp}_{suffix}.XPT"
        content, meta = download_xpt(url, dest)
        df = read_xpt_bytes(content)
        meta.update({"cycle": cycle, "component": comp, "rows": int(len(df)), "columns": int(df.shape[1])})
        manifest.append(meta)
        frames[comp] = df

    demo = frames["DEMO"].copy()
    dpq = frames["DPQ"].copy()
    slq = frames["SLQ"].copy()
    for df in (demo, dpq, slq):
        df["SEQN"] = numeric(df["SEQN"]).astype("Int64")

    keep_demo = [c for c in ["SEQN", "RIDAGEYR", "RIAGENDR", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"] if c in demo.columns]
    keep_dpq = ["SEQN"] + sorted({v for _, v, _, _ in NODE_SPECS if v.startswith("DPQ")})
    keep_slq = ["SEQN"] + sorted({v for _, v, _, _ in NODE_SPECS if v.startswith("SL")})

    merged = demo[keep_demo].merge(dpq[keep_dpq], on="SEQN", how="left", validate="one_to_one")
    merged = merged.merge(slq[keep_slq], on="SEQN", how="left", validate="one_to_one")
    merged["RIDAGEYR"] = numeric(merged["RIDAGEYR"])
    eligible = merged.loc[merged["RIDAGEYR"] >= 65].copy()
    eligible["cycle"] = cycle

    for name, raw_var, domain, rule in NODE_SPECS:
        eligible[name] = recode_node(eligible[raw_var], rule)

    keep_ready = ["SEQN", "cycle", "RIDAGEYR", "RIAGENDR", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"] + NODES
    ready_all = eligible[keep_ready].copy()
    complete = ready_all.dropna(subset=NODES).copy()
    for n in NODES:
        complete[n] = complete[n].astype(int)

    audit_rows = []
    weights = numeric(ready_all["WTMEC2YR"]) if "WTMEC2YR" in ready_all else pd.Series(1.0, index=ready_all.index)
    cc_weights = numeric(complete["WTMEC2YR"]) if "WTMEC2YR" in complete else pd.Series(1.0, index=complete.index)
    for n in NODES:
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
            "domain": DOMAINS[n],
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
    audit = pd.DataFrame(audit_rows)
    return ready_all, complete, audit


def comb_log(p: int, k: int) -> float:
    if k <= 0 or k >= p:
        return 0.0
    return float(gammaln(p + 1) - gammaln(k + 1) - gammaln(p - k + 1))


def fit_logit_l1(X: np.ndarray, y: np.ndarray, C: float) -> tuple[np.ndarray, float, np.ndarray]:
    if np.unique(y).size < 2:
        p0 = (y.sum() + 0.5) / (len(y) + 1.0)
        intercept = math.log(p0 / (1 - p0))
        return np.zeros(X.shape[1]), intercept, np.full(len(y), p0)
    m = LogisticRegression(
        penalty="l1", solver="liblinear", C=float(C), fit_intercept=True,
        max_iter=5000, tol=1e-8, random_state=BOOT_SEED,
    )
    m.fit(X, y)
    coef = m.coef_[0].astype(float)
    intercept = float(m.intercept_[0])
    prob = expit(intercept + X @ coef)
    return coef, intercept, prob


def select_node_model(X: np.ndarray, y: np.ndarray) -> dict:
    p = X.shape[1]
    if np.unique(y).size < 2:
        coef, intercept, prob = fit_logit_l1(X, y, C_GRID[0])
        return {"coef": coef, "intercept": intercept, "C": float(C_GRID[0]), "ebic": np.nan, "k": 0, "prob": prob}
    candidates = []
    for C in C_GRID:
        coef, intercept, prob = fit_logit_l1(X, y, C)
        prob = np.clip(prob, 1e-12, 1 - 1e-12)
        ll = float(np.sum(y * np.log(prob) + (1 - y) * np.log(1 - prob)))
        k = int(np.sum(np.abs(coef) > ZERO_TOL))
        df = k + 1
        ebic = -2 * ll + df * math.log(len(y)) + 2 * GAMMA * comb_log(p, k)
        candidates.append((ebic, float(C), k, coef, intercept, prob))
    candidates.sort(key=lambda z: (z[0], z[1]))
    ebic, C, k, coef, intercept, prob = candidates[0]
    return {"coef": coef, "intercept": intercept, "C": C, "ebic": float(ebic), "k": k, "prob": prob}


def symmetrize(directional: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = directional.shape[0]
    J = np.zeros((p, p), dtype=float)
    sign_conflict = np.zeros((p, p), dtype=bool)
    for i in range(p):
        for j in range(i + 1, p):
            a, b = directional[i, j], directional[j, i]
            if abs(a) > ZERO_TOL and abs(b) > ZERO_TOL:
                J[i, j] = J[j, i] = (a + b) / 2.0
                sign_conflict[i, j] = sign_conflict[j, i] = np.sign(a) != np.sign(b)
    return J, sign_conflict


def fit_ising(df: pd.DataFrame, fixed_C: np.ndarray | None = None) -> dict:
    Xall = df[NODES].to_numpy(dtype=float)
    p = Xall.shape[1]
    directional = np.zeros((p, p), dtype=float)
    intercepts = np.zeros(p, dtype=float)
    selected_C = np.zeros(p, dtype=float)
    ebics = np.full(p, np.nan)
    ks = np.zeros(p, dtype=int)
    predictability = np.full(p, np.nan)

    for i in range(p):
        idx = [j for j in range(p) if j != i]
        X = Xall[:, idx]
        y = Xall[:, i].astype(int)
        if fixed_C is None:
            r = select_node_model(X, y)
        else:
            coef, intercept, prob = fit_logit_l1(X, y, float(fixed_C[i]))
            r = {"coef": coef, "intercept": intercept, "C": float(fixed_C[i]), "ebic": np.nan, "k": int(np.sum(np.abs(coef) > ZERO_TOL)), "prob": prob}
        directional[i, idx] = r["coef"]
        intercepts[i] = r["intercept"]
        selected_C[i] = r["C"]
        ebics[i] = r["ebic"]
        ks[i] = r["k"]
        prob = np.asarray(r["prob"])
        if np.any(y == 1) and np.any(y == 0):
            predictability[i] = float(prob[y == 1].mean() - prob[y == 0].mean())

    J, conflicts = symmetrize(directional)
    return {
        "J": J, "h": intercepts, "directional": directional, "C": selected_C,
        "ebic": ebics, "k": ks, "predictability": predictability, "sign_conflict": conflicts,
    }


STATES = np.array(list(product([0.0, 1.0], repeat=P)), dtype=float)
DEP_IDX = np.array([i for i, n in enumerate(NODES) if DOMAINS[n] == "depression"], dtype=int)
SLP_IDX = np.array([i for i, n in enumerate(NODES) if DOMAINS[n] == "sleep"], dtype=int)


def ising_prob(J: np.ndarray, h: np.ndarray) -> np.ndarray:
    energy = STATES @ h + 0.5 * np.sum((STATES @ J) * STATES, axis=1)
    return np.exp(energy - logsumexp(energy))


def perturbations(J: np.ndarray, h: np.ndarray) -> pd.DataFrame:
    prob = ising_prob(J, h)
    total = STATES.sum(axis=1)
    dep_b = STATES[:, DEP_IDX].sum(axis=1)
    slp_b = STATES[:, SLP_IDX].sum(axis=1)
    base_total = float(prob @ total)
    base_dep = float(prob @ dep_b)
    base_slp = float(prob @ slp_b)
    rows = []
    for j, node in enumerate(NODES):
        mask = STATES[:, j] == 0
        pc = prob[mask]
        pc = pc / pc.sum()
        cond_total = float(pc @ total[mask])
        cond_dep = float(pc @ dep_b[mask])
        cond_slp = float(pc @ slp_b[mask])
        rows.append({
            "node": node,
            "domain": DOMAINS[node],
            "baseline_total_burden": base_total,
            "conditional_total_burden_Xj0": cond_total,
            "R_total": base_total - cond_total,
            "R_depression": base_dep - cond_dep,
            "R_sleep": base_slp - cond_slp,
            "R_other_domain": (base_dep - cond_dep) if DOMAINS[node] == "sleep" else (base_slp - cond_slp),
        })
    return pd.DataFrame(rows)


def edge_table(fit: dict, cycle: str) -> pd.DataFrame:
    rows = []
    J = fit["J"]
    D = fit["directional"]
    C = fit["sign_conflict"]
    for i in range(P):
        for j in range(i + 1, P):
            rows.append({
                "cycle": cycle,
                "node_i": NODES[i], "node_j": NODES[j],
                "domain_i": DOMAINS[NODES[i]], "domain_j": DOMAINS[NODES[j]],
                "bridge": DOMAINS[NODES[i]] != DOMAINS[NODES[j]],
                "coef_i_given_j": D[i, j], "coef_j_given_i": D[j, i],
                "and_included": bool(abs(J[i, j]) > ZERO_TOL),
                "sign_conflict": bool(C[i, j]),
                "edge_weight": J[i, j],
            })
    return pd.DataFrame(rows)


def node_table(fit: dict, cycle: str) -> pd.DataFrame:
    J = fit["J"]
    rows = []
    for i, n in enumerate(NODES):
        other = [j for j in range(P) if DOMAINS[NODES[j]] != DOMAINS[n]]
        rows.append({
            "cycle": cycle, "node": n, "domain": DOMAINS[n],
            "strength": float(np.abs(J[i]).sum()),
            "bridge_strength": float(np.abs(J[i, other]).sum()),
            "predictability_tjur": fit["predictability"][i],
            "selected_C": fit["C"][i], "selected_nonzero_k": int(fit["k"][i]), "selected_EBIC": fit["ebic"][i],
            "intercept_h": fit["h"][i],
        })
    return pd.DataFrame(rows)


def bootstrap_derivation(df: pd.DataFrame, selected_C: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOT_SEED)
    n = len(df)
    Rs = np.zeros((BOOT_B, P), dtype=float)
    Rother = np.zeros((BOOT_B, P), dtype=float)
    edges = np.zeros((BOOT_B, P, P), dtype=float)
    strengths = np.zeros((BOOT_B, P), dtype=float)
    for b in range(BOOT_B):
        idx = rng.integers(0, n, size=n)
        boot = df.iloc[idx].reset_index(drop=True)
        f = fit_ising(boot, fixed_C=selected_C)
        ptab = perturbations(f["J"], f["h"])
        Rs[b, :] = ptab["R_total"].to_numpy()
        Rother[b, :] = ptab["R_other_domain"].to_numpy()
        edges[b] = f["J"]
        strengths[b] = np.abs(f["J"]).sum(axis=1)
        if (b + 1) % 50 == 0:
            print(f"bootstrap {b+1}/{BOOT_B}", flush=True)

    node_rows = []
    for i, node in enumerate(NODES):
        node_rows.append({
            "node": node, "domain": DOMAINS[node],
            "R_total_boot_mean": float(Rs[:, i].mean()),
            "R_total_boot_median": float(np.median(Rs[:, i])),
            "R_total_boot_q025": float(np.quantile(Rs[:, i], 0.025)),
            "R_total_boot_q975": float(np.quantile(Rs[:, i], 0.975)),
            "P_R_total_gt0": float(np.mean(Rs[:, i] > 0)),
            "R_other_boot_mean": float(Rother[:, i].mean()),
            "R_other_boot_q025": float(np.quantile(Rother[:, i], 0.025)),
            "R_other_boot_q975": float(np.quantile(Rother[:, i], 0.975)),
            "P_R_other_gt0": float(np.mean(Rother[:, i] > 0)),
            "strength_boot_mean": float(strengths[:, i].mean()),
            "strength_boot_q025": float(np.quantile(strengths[:, i], 0.025)),
            "strength_boot_q975": float(np.quantile(strengths[:, i], 0.975)),
        })
    edge_rows = []
    for i in range(P):
        for j in range(i + 1, P):
            v = edges[:, i, j]
            inc = np.abs(v) > ZERO_TOL
            edge_rows.append({
                "node_i": NODES[i], "node_j": NODES[j],
                "bridge": DOMAINS[NODES[i]] != DOMAINS[NODES[j]],
                "edge_inclusion_probability": float(inc.mean()),
                "edge_weight_boot_mean": float(v.mean()),
                "edge_weight_boot_q025": float(np.quantile(v, 0.025)),
                "edge_weight_boot_q975": float(np.quantile(v, 0.975)),
                "positive_probability": float(np.mean(v > 0)),
                "negative_probability": float(np.mean(v < 0)),
            })
    return pd.DataFrame(node_rows), pd.DataFrame(edge_rows)


def safe_spearman(a, b) -> float:
    r = spearmanr(np.asarray(a, float), np.asarray(b, float), nan_policy="omit")
    return float(r.statistic) if np.isfinite(r.statistic) else np.nan


def qualification(edge_d: pd.DataFrame, edge_v: pd.DataFrame, node_d: pd.DataFrame, node_v: pd.DataFrame,
                  pert_d: pd.DataFrame, pert_v: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    e = edge_d[["node_i", "node_j", "bridge", "edge_weight", "and_included"]].rename(
        columns={"edge_weight":"derivation_weight", "and_included":"derivation_included"})
    e = e.merge(edge_v[["node_i", "node_j", "edge_weight", "and_included"]].rename(
        columns={"edge_weight":"validation_weight", "and_included":"validation_included"}), on=["node_i", "node_j"], how="inner")
    e["same_sign"] = np.sign(e.derivation_weight) == np.sign(e.validation_weight)
    e["retained_same_sign"] = e.derivation_included & e.validation_included & e.same_sign
    de = e[e.derivation_included]
    both = e[e.derivation_included & e.validation_included]
    bridge = e[e.bridge]

    nd = node_d.set_index("node")
    nv = node_v.set_index("node")
    pdx = pert_d.set_index("node")
    pvx = pert_v.set_index("node")
    top3_d = set(pdx.R_total.nlargest(3).index)
    top3_v = set(pvx.R_total.nlargest(3).index)
    top3_j = len(top3_d & top3_v) / len(top3_d | top3_v)

    metrics = {
        "derivation_edges_n": int(edge_d.and_included.sum()),
        "validation_edges_n": int(edge_v.and_included.sum()),
        "derivation_edge_retention_rate": float((de.validation_included).mean()) if len(de) else np.nan,
        "derivation_edge_same_sign_retention_rate": float((de.validation_included & de.same_sign).mean()) if len(de) else np.nan,
        "sign_concordance_among_retained_edges": float(both.same_sign.mean()) if len(both) else np.nan,
        "all_78_edge_weight_spearman": safe_spearman(e.derivation_weight, e.validation_weight),
        "bridge_edge_weight_spearman": safe_spearman(bridge.derivation_weight, bridge.validation_weight),
        "node_strength_spearman": safe_spearman(nd.loc[NODES, "strength"], nv.loc[NODES, "strength"]),
        "bridge_strength_spearman": safe_spearman(nd.loc[NODES, "bridge_strength"], nv.loc[NODES, "bridge_strength"]),
        "perturbation_R_total_rank_spearman": safe_spearman(pdx.loc[NODES, "R_total"], pvx.loc[NODES, "R_total"]),
        "perturbation_R_other_rank_spearman": safe_spearman(pdx.loc[NODES, "R_other_domain"], pvx.loc[NODES, "R_other_domain"]),
        "perturbation_R_total_direction_concordance": float((np.sign(pdx.loc[NODES, "R_total"]) == np.sign(pvx.loc[NODES, "R_total"])).mean()),
        "top3_perturbation_jaccard": float(top3_j),
        "top3_derivation": sorted(top3_d),
        "top3_validation": sorted(top3_v),
    }
    return e, metrics


def json_safe(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, dict): return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, list): return [json_safe(v) for v in x]
    return x


def main():
    manifest = []
    ready_all = {}
    complete = {}
    audits = []

    for cycle, cfg in CYCLES.items():
        print(f"Downloading/building {cycle}", flush=True)
        ra, cc, au = build_cycle(cycle, cfg, manifest)
        ready_all[cycle] = ra
        complete[cycle] = cc
        audits.append(au)
        ra.to_csv(OUT / f"eligible65_with_missing_{cycle.replace('-', '_')}.csv", index=False)
        cc.to_csv(OUT / f"analysis_ready_{cycle.replace('-', '_')}.csv", index=False)

    audit = pd.concat(audits, ignore_index=True)
    audit.to_csv(OUT / "missingness_prevalence.csv", index=False)

    flow_rows = []
    gates = {}
    for cycle in CYCLES:
        au = audit[audit.cycle == cycle]
        eligible_n = int(ready_all[cycle].shape[0])
        cc_n = int(complete[cycle].shape[0])
        retention = cc_n / eligible_n if eligible_n else np.nan
        max_miss = float(au.missing_fraction.max())
        gate = bool(max_miss <= 0.15 and retention >= 0.70)
        gates[cycle] = gate
        flow_rows.append({
            "cycle": cycle, "eligible_age65_n": eligible_n, "complete_case_n": cc_n,
            "complete_case_retention": retention, "max_node_missing_fraction": max_miss,
            "missingness_gate_max15pct": bool(max_miss <= 0.15),
            "retention_gate_min70pct": bool(retention >= 0.70),
            "phase3_analysis_gate": gate,
        })
    flow = pd.DataFrame(flow_rows)
    flow.to_csv(OUT / "cohort_flow.csv", index=False)

    protocol = {
        "title": TITLE, "freeze_date": RUN_DATE,
        "research_question": "Among adults aged >=65 years, what conditional dependency structure links depressive and sleep-related symptoms, and which symptom-level virtual perturbations produce the largest model-implied reduction in adverse symptom burden while remaining reproducible across an independent NHANES cycle?",
        "derivation_cycle": "2015-2016", "validation_cycle": "2017-2018", "age_min": 65,
        "nodes": [{"node":n,"raw_variable":v,"domain":d,"rule":r} for n,v,d,r in NODE_SPECS],
        "missingness_primary": "complete-case across 13 nodes",
        "stop_rules": {"max_node_missing_fraction": 0.15, "min_complete_case_retention": 0.70},
        "descriptive_weight": "WTMEC2YR",
        "primary_network": "unweighted pairwise Ising via nodewise L1 logistic pseudolikelihood",
        "ebic_gamma": GAMMA, "C_grid": C_GRID.tolist(), "symmetrization": "AND; edge weight=mean directional coefficients",
        "validation": "untouched independent 2017-2018 cycle",
        "perturbation": "exact 2^13-state enumeration; condition X_j=0; not a causal do-intervention",
        "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "regularization": "fixed at derivation-selected nodewise C"},
        "credibility_rule": "P_bootstrap(R_total>0)>=0.95 AND validation R_total>0",
        "abstention_categories": ["credible", "directionally reproducible but uncertain", "non-replicating", "not estimable"],
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    protocol["protocol_sha256"] = hashlib.sha256(canonical).hexdigest()
    (OUT / "protocol_lock.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not all(gates.values()):
        summary = {"status":"ABSTAINED_NOT_ESTIMABLE", "reason":"Prespecified Phase 1-2 missingness/retention gate failed", "gates":gates}
        (OUT / "phase3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        der = complete["2015-2016"].reset_index(drop=True)
        val = complete["2017-2018"].reset_index(drop=True)
        print(f"Fitting derivation Ising n={len(der)}", flush=True)
        fit_d = fit_ising(der)
        print(f"Fitting validation Ising n={len(val)}", flush=True)
        fit_v = fit_ising(val)
        edge_d = edge_table(fit_d, "2015-2016")
        edge_v = edge_table(fit_v, "2017-2018")
        node_d = node_table(fit_d, "2015-2016")
        node_v = node_table(fit_v, "2017-2018")
        pert_d = perturbations(fit_d["J"], fit_d["h"]); pert_d.insert(0, "cycle", "2015-2016")
        pert_v = perturbations(fit_v["J"], fit_v["h"]); pert_v.insert(0, "cycle", "2017-2018")

        edge_d.to_csv(OUT / "network_edges_derivation.csv", index=False)
        edge_v.to_csv(OUT / "network_edges_validation.csv", index=False)
        node_d.to_csv(OUT / "node_metrics_derivation.csv", index=False)
        node_v.to_csv(OUT / "node_metrics_validation.csv", index=False)
        pert_d.to_csv(OUT / "perturbation_derivation.csv", index=False)
        pert_v.to_csv(OUT / "perturbation_validation.csv", index=False)
        np.save(OUT / "J_derivation.npy", fit_d["J"])
        np.save(OUT / "h_derivation.npy", fit_d["h"])
        np.save(OUT / "J_validation.npy", fit_v["J"])
        np.save(OUT / "h_validation.npy", fit_v["h"])

        print("Running derivation bootstrap", flush=True)
        boot_node, boot_edge = bootstrap_derivation(der, fit_d["C"])
        boot_node.to_csv(OUT / "bootstrap_node_results.csv", index=False)
        boot_edge.to_csv(OUT / "bootstrap_edge_results.csv", index=False)

        eval_edges, qmetrics = qualification(edge_d, edge_v, node_d, node_v, pert_d, pert_v)
        eval_edges.to_csv(OUT / "temporal_edge_validation.csv", index=False)

        credibility = pert_d[["node", "domain", "R_total", "R_other_domain"]].rename(columns={"R_total":"R_total_derivation", "R_other_domain":"R_other_derivation"})
        credibility = credibility.merge(pert_v[["node", "R_total", "R_other_domain"]].rename(columns={"R_total":"R_total_validation", "R_other_domain":"R_other_validation"}), on="node")
        credibility = credibility.merge(boot_node[["node", "P_R_total_gt0", "R_total_boot_q025", "R_total_boot_q975", "P_R_other_gt0", "R_other_boot_q025", "R_other_boot_q975"]], on="node")
        cats = []
        for r in credibility.itertuples():
            if r.P_R_total_gt0 >= 0.95 and r.R_total_validation > 0:
                cats.append("credible")
            elif r.R_total_derivation > 0 and r.R_total_validation > 0:
                cats.append("directionally reproducible but uncertain")
            else:
                cats.append("non-replicating")
        credibility["credibility_category"] = cats
        credibility.to_csv(OUT / "perturbation_credibility.csv", index=False)

        summary = {
            "status":"COMPLETE", "gates":gates,
            "derivation_n": int(len(der)), "validation_n": int(len(val)),
            "qualification_metrics": qmetrics,
            "credible_nodes": credibility.loc[credibility.credibility_category == "credible", "node"].tolist(),
            "nonreplicating_nodes": credibility.loc[credibility.credibility_category == "non-replicating", "node"].tolist(),
        }
        (OUT / "phase3_summary.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")

    pd.concat([complete[c] for c in CYCLES], ignore_index=True).to_csv(OUT / "analysis_ready_combined.csv", index=False)

    hashes = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "results_lock.json":
            hashes[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    results_lock = {
        "title": TITLE, "run_date": RUN_DATE, "protocol_sha256": protocol["protocol_sha256"],
        "files": hashes,
    }
    (OUT / "results_lock.json").write_text(json.dumps(results_lock, indent=2), encoding="utf-8")
    print(json.dumps(json_safe({"flow": flow.to_dict(orient="records"), "gates":gates}), indent=2), flush=True)


if __name__ == "__main__":
    main()
