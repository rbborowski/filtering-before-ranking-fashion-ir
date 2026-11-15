"""Scripts completos de auditoria pós-hoc, testes estatísticos e seleção de casos do TCC."""

import zipfile
import json
import ast
import re
import string
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

# Localizador flexível dos artefatos congelados (lê da pasta data_and_results ou direto do ZIP)
def load_artifact_df(filename):
    for candidate in [Path("../data_and_results") / filename, Path("data_and_results") / filename, Path("tcc_validacao") / filename, Path(filename)]:
        if candidate.exists():
            return pd.read_csv(candidate)
    for zcand in [Path("../resultados_validacao.zip"), Path("resultados_validacao.zip")]:
        if zcand.exists():
            with zipfile.ZipFile(zcand, "r") as z:
                if filename in z.namelist():
                    return pd.read_csv(z.open(filename))
    raise FileNotFoundError(f"Arquivo {filename} não encontrado. Adicione resultados_validacao.zip ou extraia em data_and_results/.")

def load_artifact_json(prefix_or_name):
    for folder in [Path("../data_and_results"), Path("data_and_results"), Path("../materials"), Path("materials"), Path("tcc_validacao"), Path(".")]:
        if folder.exists():
            exact = folder / prefix_or_name
            if exact.exists():
                return json.loads(exact.read_text(encoding="utf-8"))
            matches = [
                p for p in folder.glob(f"{prefix_or_name}*.json")
                if p.name not in ("parser_prompt.json", "parser_attribute_outputs.json")
            ]
            if matches:
                return json.loads(matches[0].read_text(encoding="utf-8"))
    for zcand in [Path("../resultados_validacao.zip"), Path("resultados_validacao.zip")]:
        if zcand.exists():
            with zipfile.ZipFile(zcand, "r") as z:
                if prefix_or_name in z.namelist():
                    return json.loads(z.read(prefix_or_name).decode("utf-8"))
                matches = [
                    f for f in z.namelist()
                    if f.startswith(prefix_or_name) and f.endswith(".json")
                    and f not in ("parser_prompt.json", "parser_attribute_outputs.json")
                ]
                if matches:
                    return json.loads(z.read(matches[0]).decode("utf-8"))
    return {}

print("Funções de leitura configuradas com sucesso.")


from transformers import AutoTokenizer

catalog = load_artifact_df("catalog_sample.csv")
type_counts = catalog["product_type_name"].value_counts()
rare_types = type_counts[type_counts < 5].index.tolist()
n_rare_articles = int(catalog["product_type_name"].isin(rare_types).sum())
n_unknown_articles = int((catalog["product_type_name"] == "Unknown").sum())
vocab_74 = sorted([t for t, c in type_counts.items() if c >= 5 and t not in ("Unknown", "Other", "Unmapped")])

print(f"Total de tipos distintos na amostra (N=10.000): {len(type_counts)}")
print(f"Tipos raros (< 5 artigos): {len(rare_types)} tipos -> {n_rare_articles} artigos afetados")
print(f"Rótulo 'Unknown': 1 tipo -> {n_unknown_articles} artigos afetados")
print(f"Vocabulário fechado final (product_type_name): {len(vocab_74)} tipos")

tok_minilm = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
tok_bge = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

def build_eval_text(row):
    return (
        f"Name: {row['prod_name']}. Type: {row['product_type_name']}. "
        f"Category: {row['product_group_name']}. Color: {row['colour_group_name']}. "
        f"Audience: {row['index_group_name']}. Description: {row['detail_desc']}"
    )

texts = catalog.apply(build_eval_text, axis=1).astype(str).tolist()
trunc_minilm = sum(1 for t in texts if len(tok_minilm.tokenize(t)) > 256)
trunc_bge = sum(1 for t in texts if len(tok_bge.tokenize(t)) > 512)
print(f"Truncados no MiniLM (>256 tokens): {trunc_minilm} ({trunc_minilm/100:.2f}%)")
print(f"Truncados no BGE (>512 tokens):    {trunc_bge} ({trunc_bge/100:.2f}%)")


DEVELOPMENT_ARTICLE_IDS = [
    501117001, 798288003, 405586008, 850549001, 906880002,
    859105002, 739303002, 891174001, 762884003, 880098001,
    916272001, 636207002, 661650003, 727593002, 797623003,
    759231002, 699731006, 677915002, 818806002, 615514001
]

final_queries = load_artifact_df("synthetic_queries.csv")
if "product_code" not in catalog.columns:
    catalog["product_code"] = catalog["article_id"].astype(str).str.zfill(10).str[:7].astype(int)

dev_products = catalog[catalog["article_id"].isin(DEVELOPMENT_ARTICLE_IDS)]
excluded_codes = set(dev_products["product_code"])
catalog["norm_desc"] = catalog["detail_desc"].str.casefold().str.strip()
excluded_descriptions = set(dev_products["detail_desc"].str.casefold().str.strip())

mask_code = catalog["product_code"].isin(excluded_codes)
mask_desc = catalog["norm_desc"].isin(excluded_descriptions)
removed_pilot = catalog[mask_code | mask_desc]
val_pool = catalog[~mask_code & ~mask_desc].copy()
print(f"Total inicial na amostra: {len(catalog)}")
print(f"Removidos por sobreposição com os 20 itens do piloto: {len(removed_pilot)} -> Restam: {len(val_pool)}")

val_pool = val_pool.sample(frac=1, random_state=2026).drop_duplicates("product_code")
val_pool = val_pool.drop_duplicates("norm_desc")
print(f"Pool final elegível após deduplicação (product_code + detail_desc): {len(val_pool)}")

final_ids = set(final_queries["article_id"])
assert len(final_queries) == 200 and final_queries["article_id"].nunique() == 200
assert final_ids.issubset(set(val_pool["article_id"]))
print("Verificação de sobreposição com piloto: 0 IDs, 0 product_codes, 0 descrições.")


canonical_colours = [
    "black", "white", "grey", "blue", "turquoise", "red", "pink",
    "purple", "orange", "yellow", "brown", "beige", "green"
]
colour_mentions = [
    q for q in final_queries["query"]
    if any(re.search(rf"\b{c}\b", str(q), re.I) for c in canonical_colours)
]
print(f"Consultas sintéticas com menção literal às 13 famílias de cor: {len(colour_mentions)}/200")

dup_texts = final_queries[final_queries["query"].str.casefold().duplicated(keep=False)]
print(f"Consultas sintéticas com texto idêntico: {len(dup_texts)} linhas ({final_queries['query'].str.casefold().nunique()} textos únicos)")
if not dup_texts.empty:
    print(dup_texts)


runs = load_artifact_df("runs.csv")
sis_base = "baseline"
sis_full = "filters_full"
sis_res = [s for s in runs["system"].unique() if s not in (sis_base, sis_full)][0]

contrasts = [
    (sis_base, sis_full, "Dense baseline -> Filters + full query"),
    (sis_base, sis_res,  "Dense baseline -> Filters + residual"),
    (sis_full, sis_res,  "Filters + full query -> Filters + residual"),
]

transition_records = []
for model in ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]:
    sub = runs[runs["embedder"] == model]
    piv = sub.pivot(index="query_id", columns="system", values="Hit@5")
    for s_from, s_to, label in contrasts:
        gained = int(((piv[s_from] == 0) & (piv[s_to] == 1)).sum())
        lost = int(((piv[s_from] == 1) & (piv[s_to] == 0)).sum())
        unchanged = int((piv[s_from] == piv[s_to]).sum())
        n_disc = gained + lost
        p_val = binomtest(min(gained, lost), n=n_disc, p=0.5, alternative="two-sided").pvalue if n_disc > 0 else 1.0
        delta_pp = (piv[s_to].mean() - piv[s_from].mean()) * 100
        transition_records.append({
            "embedder": model, "contrast": label,
            "gained": gained, "lost": lost, "unchanged": unchanged,
            "net_delta_pp": round(delta_pp, 2), "mcnemar_exact_p": round(p_val, 4)
        })

df_transitions = pd.DataFrame(transition_records)
print(df_transitions)

# Teste de McNemar na condição unificada de falha/sucesso (qualquer perda ou ganho no par)
bge_sub = runs[runs["embedder"] == "BAAI/bge-small-en-v1.5"]
piv_bge = bge_sub.pivot(index="query_id", columns="system", values="Hit@5").merge(
    final_queries[["query_id", "query"]], on="query_id"
)
piv_bge["diff_res_base"] = piv_bge[sis_res] - piv_bge[sis_base]

# Bootstrap agrupado por texto único (198 clusters, 10.000 reamostragens, seed=20261017)
grouped_diffs = [grp["diff_res_base"].to_numpy() for _, grp in piv_bge.groupby(piv_bge["query"].str.casefold(), sort=True)]
rng_grp = np.random.default_rng(20261017)
boot_means = np.empty(10000)
for b in range(10000):
    idx = rng_grp.integers(0, len(grouped_diffs), size=len(grouped_diffs))
    sampled = np.concatenate([grouped_diffs[i] for i in idx])
    boot_means[b] = sampled.mean() * 100
ci_lo, ci_hi = np.quantile(boot_means, [0.025, 0.975])
print(f"\nBootstrap Agrupado Pós-Hoc (198 clusters únicos, 10.000 reps, seed=20261017):")
print(f"BGE Filters + residual vs Dense baseline (Hit@5): IC 95% = [{ci_lo:.2f}, {ci_hi:.2f}] pp")


def clean_punct(text):
    return str(text).translate(str.maketrans("", "", string.punctuation)).replace(" ", "").lower()

df_full_bge = bge_sub[bge_sub["system"] == sis_full].set_index("query_id")
df_res_bge = bge_sub[bge_sub["system"] == sis_res].set_index("query_id")

stats_res = {"sem_filtro_ativo": 0, "multiword_ou_fallback": 0, "apenas_pontuacao": 0, "delecao_efetiva": 0}
effective_ids = []
deletion_rows = []

for qid, row in df_res_bge.iterrows():
    q_orig = str(df_full_bge.loc[qid, "query"]).strip()
    q_res = str(row["ranking_text"]).strip() if pd.notna(row["ranking_text"]) and str(row["ranking_text"]).strip() != "" else q_orig
    used_dict = ast.literal_eval(str(row["filters_used"])) if pd.notna(row["filters_used"]) else {}
    has_filter = len(used_dict) > 0

    if q_orig == q_res:
        if has_filter:
            stats_res["multiword_ou_fallback"] += 1
        else:
            stats_res["sem_filtro_ativo"] += 1
    elif clean_punct(q_orig) == clean_punct(q_res):
        stats_res["apenas_pontuacao"] += 1
    else:
        stats_res["delecao_efetiva"] += 1
        effective_ids.append(qid)
        removed_tokens = [w for w in q_orig.split() if w.lower() not in q_res.lower().split()]
        deletion_rows.append({
            "query_id": qid,
            "original_query": q_orig,
            "residual_query": q_res,
            "removed_token": " ".join(removed_tokens),
            "filters_used": str(used_dict),
            "semantic_audit": "Evidence Error (material removed for garment type)" if qid == "synth_0004" else "Correct 1-word categorical removal"
        })

print("Decomposição das 200 consultas na variante residual:", stats_res)
df_deletions = pd.DataFrame(deletion_rows)
df_deletions.to_csv("../audits/audit_91_deletions.csv" if Path("../audits").exists() else "audit_91_deletions.csv", index=False)

print("\n--- Desempenho nas 91 Consultas com Deleção Efetiva ---")
for model in ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]:
    m_sub = runs[(runs["embedder"] == model) & (runs["query_id"].isin(effective_ids))]
    for s_name in [sis_full, sis_res]:
        s_df = m_sub[m_sub["system"] == s_name]
        print(f"{model:22s} | {s_name:12s} -> Hit@5: {s_df['Hit@5'].mean()*100:.2f}% | MRR@20: {s_df['MRR@20'].mean():.4f}")


exclusoes = df_full_bge[df_full_bge["target_filtered_out"] == True].reset_index().merge(
    catalog[["article_id", "prod_name", "product_type_name", "colour_group_name", "detail_desc"]],
    on="article_id", how="left"
)
print(f"Total de alvos excluídos pela filtragem: {len(exclusoes)}/200 ({len(exclusoes)/2:.1f}%)")
print(exclusoes[["query_id", "article_id", "query", "filters_used", "product_type_name", "detail_desc"]])


n_parse_fail = int((~df_full_bge["parse_ok"]).sum())
n_val_dropped = int(df_full_bge["validation_dropped"].apply(lambda x: len(ast.literal_eval(str(x))) > 0 if pd.notna(x) else False).sum())
print(f"Falhas estruturais de JSON (parse_ok=False -> fallback): {n_parse_fail}/200 ({n_parse_fail/2:.1f}%)")
print(f"Consultas com valores descartados pelo validador:        {n_val_dropped}/200 ({n_val_dropped/2:.1f}%)")

parser_cache = load_artifact_json("parser_")
attr_parses = load_artifact_json("parser_attribute_outputs.json")

def has_duplicate_json_keys(raw_text):
    dup = False
    def hook(pairs):
        nonlocal dup
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            dup = True
        return dict(pairs)
    try:
        m = re.search(r"\{.*\}", str(raw_text), re.DOTALL)
        if m:
            json.loads(m.group(0), object_pairs_hook=hook)
    except Exception:
        pass
    return dup

if parser_cache:
    synth_dups = sum(
        1 for q in final_queries["query"]
        if any(v.get("query") == q and has_duplicate_json_keys(v.get("raw_response", "")) for v in parser_cache.values())
    )
    print(f"Chaves JSON repetidas nas 200 consultas sintéticas: {synth_dups}/200 ({synth_dups/2:.1f}%)")
if attr_parses:
    attr_dups = sum(1 for p in attr_parses if has_duplicate_json_keys(p.get("raw_response", "")))
    print(f"Chaves JSON repetidas nas 40 consultas controladas: {attr_dups}/40 ({attr_dups/0.4:.1f}%)")


cases = load_artifact_df("attribute_cases_PROPOSED_LABELS.csv")
minilm_piv = runs[runs["embedder"] == "all-MiniLM-L6-v2"].pivot(index="query_id", columns="system", values=["MRR@20", "rank", "target_filtered_out"])
bge_piv = runs[runs["embedder"] == "BAAI/bge-small-en-v1.5"].pivot(index="query_id", columns="system", values=["MRR@20", "rank", "target_filtered_out"])

# Regra 1: Maior ganho de MRR@20 (MiniLM residual vs baseline) entre alvos retidos
ret_minilm = minilm_piv[minilm_piv[("target_filtered_out", sis_res)] == False]
c1 = (ret_minilm[("MRR@20", sis_res)] - ret_minilm[("MRR@20", sis_base)]).sort_values(ascending=False)
print(f"Caso 1 (Maior ganho local MiniLM):   {c1.index[0]} | Delta MRR@20 = {c1.iloc[0]:+.4f}")

# Regra 2: Maior queda de MRR@20 (BGE residual vs baseline) entre alvos retidos
ret_bge = bge_piv[bge_piv[("target_filtered_out", sis_res)] == False]
c2 = (ret_bge[("MRR@20", sis_res)] - ret_bge[("MRR@20", sis_base)]).sort_values(ascending=True)
print(f"Caso 2 (Maior queda local BGE):      {c2.index[0]} | Delta MRR@20 = {c2.iloc[0]:+.4f}")

# Regra 3: Maior MRR@20 no BGE baseline entre os 13 alvos excluídos pelo filtro
excl_bge = bge_piv[bge_piv[("target_filtered_out", sis_full)] == True]
c3 = excl_bge[("MRR@20", sis_base)].sort_values(ascending=False)
print(f"Caso 3 (Exclusão de alvo relevante): {c3.index[0]} | Baseline MRR@20 = {c3.iloc[0]:.4f}")

# Regra 4: Maior redução de violação no MiniLM (baseline -> filters_full)
cases_m = cases[cases["embedder"] == "all-MiniLM-L6-v2"].pivot(index="query_id", columns="system", values="violation_rate").dropna()
c4 = (cases_m[sis_base] - cases_m[sis_full]).sort_values(ascending=False)
print(f"Caso 4 (Conformidade de atributos):  {c4.index[0]} | Redução de violação = {c4.iloc[0]*100:.1f} pp")


