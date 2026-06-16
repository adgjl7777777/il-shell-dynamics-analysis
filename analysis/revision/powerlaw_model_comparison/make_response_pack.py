#!/usr/bin/env python3
"""Build compact R2-1 response artifacts from model-comparison outputs.

This script reads only outputs generated under the revision folder and writes
response-ready notes/tables under the revision folder. It does not modify the
manuscript source.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import datetime as dt

REV = Path('/nas_2/transcendence/revision')
ANA = REV / 'analysis' / 'powerlaw_model_comparison'
OUT = ANA / 'outputs'
NOTES = REV / 'notes' / '01_r2_1_distribution_statistics'
RESP = REV / 'notes' / '06_response_package'

ALL_CSV = OUT / 'model_comparison_all.csv'
COUNTS_CSV = OUT / 'model_comparison_counts.csv'
COMPACT_CSV = OUT / 'r2_1_best_model_compact_for_si.csv'
PACK_MD = NOTES / 'r2_1_powerlaw_rescue_pack.md'

MODEL_LABELS = {
    'exponential_conditional': 'conditional exponential',
    'weibull_conditional': 'conditional Weibull / stretched exponential',
    'pareto_power_law': 'Pareto power law',
    'tempered_power_law': 'tempered power law',
    'biexponential_conditional': 'conditional biexponential mixture',
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def ffloat(x: str, default=float('nan')) -> float:
    try:
        return float(x)
    except Exception:
        return default


def fmt_num(x: float) -> str:
    if x != x:
        return 'NA'
    ax = abs(x)
    if ax >= 1000:
        return f'{x:.3g}'
    if ax >= 10:
        return f'{x:.1f}'
    if ax >= 1:
        return f'{x:.2f}'
    return f'{x:.3f}'


def best_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        if r.get('ok') == 'True':
            key = (r['dataset'], r['window'], r['anion'], r['T'], r['state'])
            groups[key].append(r)

    compact = []
    for key, vals in sorted(groups.items()):
        vals_aic = sorted(vals, key=lambda r: ffloat(r['AIC']))
        vals_bic = sorted(vals, key=lambda r: ffloat(r['BIC']))
        best = vals_aic[0]
        second = vals_aic[1] if len(vals_aic) > 1 else None
        by_model = {r['model']: r for r in vals}
        exp = by_model.get('exponential_conditional')
        pareto = by_model.get('pareto_power_law')
        tempered = by_model.get('tempered_power_law')
        compact.append({
            'dataset': key[0],
            'window': key[1],
            'anion': key[2],
            'T': key[3],
            'state': key[4],
            'n': best['n'],
            'xmin': fmt_num(ffloat(best['xmin'])),
            'best_AIC_model': best['model'],
            'best_AIC_model_label': MODEL_LABELS.get(best['model'], best['model']),
            'best_BIC_model': vals_bic[0]['model'],
            'second_AIC_model': second['model'] if second else '',
            'delta_AIC_second_minus_best': fmt_num(ffloat(second['AIC']) - ffloat(best['AIC'])) if second else '',
            'delta_AIC_exponential_minus_best': fmt_num(ffloat(exp['AIC']) - ffloat(best['AIC'])) if exp else 'NA',
            'delta_AIC_pareto_minus_best': fmt_num(ffloat(pareto['AIC']) - ffloat(best['AIC'])) if pareto else 'NA',
            'delta_AIC_tempered_minus_best': fmt_num(ffloat(tempered['AIC']) - ffloat(best['AIC'])) if tempered else 'NA',
        })
    return compact


def write_compact_csv(rows: list[dict[str, str]]) -> None:
    fields = [
        'dataset', 'window', 'anion', 'T', 'state', 'n', 'xmin',
        'best_AIC_model', 'best_AIC_model_label', 'best_BIC_model',
        'second_AIC_model', 'delta_AIC_second_minus_best',
        'delta_AIC_exponential_minus_best', 'delta_AIC_pareto_minus_best',
        'delta_AIC_tempered_minus_best',
    ]
    with COMPACT_CSV.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})


def counts_by_group(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, int]]:
    out: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        out[(r['dataset'], r['window'])][r['best_AIC_model']] += 1
    return out


def median_delta(rows: list[dict[str, str]], dataset: str, window: str, col: str) -> str:
    vals = [ffloat(r[col]) for r in rows if r['dataset'] == dataset and r['window'] == window and r[col] != 'NA']
    vals = sorted(v for v in vals if v == v)
    if not vals:
        return 'NA'
    n = len(vals)
    med = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    return fmt_num(med)


def make_pack(count_rows: list[dict[str, str]], compact: list[dict[str, str]]) -> str:
    today = dt.date.today().isoformat()
    c = counts_by_group(compact)
    def model_counts(dataset: str, window: str) -> str:
        items = sorted(c[(dataset, window)].items(), key=lambda kv: (-kv[1], kv[0]))
        return ', '.join(f'{MODEL_LABELS.get(k, k)}: {v}' for k, v in items)

    full_pair_exp = median_delta(compact, 'pair_survival', 'full', 'delta_AIC_exponential_minus_best')
    tail_pair_exp = median_delta(compact, 'pair_survival', 'tail_q75', 'delta_AIC_exponential_minus_best')
    inter_tail_exp = median_delta(compact, 'shell_change_interevent_total', 'tail_q75', 'delta_AIC_exponential_minus_best')

    return rf"""# R2-1 power-law rescue pack

Date: {today}

Scope: Reviewer 2 Comment 1, evidence for power-law-like/heavy-tailed behavior. This pack summarizes the first likelihood/AIC/BIC model comparison and proposes safe revision language. It does not edit the manuscript source.

## Cold read

The data are **not fatal**, but they do not support a strong claim that every full pair-survival distribution is a pure or unique power law. The defensible claim is narrower and safer: Li-anion pair survival and shell-change timing are non-single-exponential, temporally heterogeneous, and often heavy-tailed over finite observation windows.

## What survived

- A single exponential is generally a poor description. For full state-resolved pair survival, the median AIC penalty for the conditional exponential relative to the best model is {full_pair_exp}. For the pair-survival upper-quartile tails, the median penalty is {tail_pair_exp}. For shell-change inter-event tails, it is {inter_tail_exp}.
- The two-state framework still has a clear role: it partitions kinetic heterogeneity into exchange-rich soft periods and longer-lived hard periods. This conclusion does not require proving an asymptotic power law.
- The inter-event tail result is helpful: for shell-change inter-event upper-quartile tails, tempered power law is the best AIC/BIC model in 6 of 12 systems.

## What should be softened

- Do not write that the pair-survival distributions **are power laws** without qualification.
- Do not frame \(\beta\) as a universal or asymptotic exponent.
- Avoid using \(R^2\) as the main evidence. It can remain as a descriptive diagnostic, but the response should lead with likelihood/AIC/BIC.

## Model-comparison snapshot

| Observable | Window | Best-AIC count summary | Safe interpretation |
|---|---|---|---|
| State-resolved pair survival | Full distribution | {model_counts('pair_survival', 'full')} | Full distributions are non-single-exponential, but not uniquely power-law. |
| State-resolved pair survival | Upper-quartile tail | {model_counts('pair_survival', 'tail_q75')} | Tail behavior remains heterogeneous; finite-window heavy-tail language is safer than pure power-law language. |
| Shell-change inter-event intervals | Full distribution | {model_counts('shell_change_interevent_total', 'full')} | Multiple relaxation-time descriptions compete; single exponential is inadequate. |
| Shell-change inter-event intervals | Upper-quartile tail | {model_counts('shell_change_interevent_total', 'tail_q75')} | Tempered power-law tails are often competitive or preferred; this supports a finite-window bursty-tail description. |

Compact SI-ready table: `{COMPACT_CSV}`

## Recommended main-text insertion

Use near the current paragraph discussing statistical caution for \(\beta\), around the Table 3/Figure 11 discussion:

> To test whether the apparent log-log regimes uniquely require a power-law description, we performed likelihood-based model comparisons using conditional exponential, conditional Weibull/stretched-exponential, Pareto power-law, tempered power-law, and conditional biexponential-mixture models over common fitting windows (Supporting Information). The comparison shows that a single exponential is strongly disfavored for most distributions, confirming non-single-exponential survival and inter-event statistics. However, pure power-law models do not uniquely dominate the full state-resolved pair-survival distributions; Weibull/stretched-exponential and biexponential-mixture descriptions are often statistically competitive or preferred. We therefore interpret the fitted \(\beta\) values as finite-window descriptors of tail heaviness rather than asymptotic power-law exponents.

## Recommended response-letter text

> We agree with the reviewer that log-log visual linearity and \(R^2\) comparisons are insufficient to establish a power-law distribution. In the revised manuscript, we therefore added a likelihood-based model comparison over common fitting windows. Specifically, we compared conditional exponential, conditional Weibull/stretched-exponential, Pareto power-law, tempered power-law, and conditional biexponential-mixture models using log-likelihood, AIC, and BIC. This analysis showed that the simple single-exponential reference is strongly disfavored for most state-resolved pair-survival and shell-change inter-event distributions. At the same time, the analysis also showed that pure power-law models do not uniquely dominate the full pair-survival distributions. We therefore revised the language throughout the manuscript to describe the fitted exponents as finite-window tail-heaviness descriptors, rather than asymptotic universal power-law exponents, and we added the model-comparison results to the Supporting Information.

## Exact manuscript actions later

1. Abstract: replace “survival times ... are described by distinct power-law-like forms” with “survival times show non-single-exponential, finite-window heavy-tail behavior”.
2. Figure 9 caption: replace “consistent with power-law-like survival behavior” with “show broad non-single-exponential decay; fitted finite-window slopes are used as descriptive tail metrics”.
3. Table 3 text: explicitly state that \(\beta\) values are descriptive and should not be interpreted as universal exponents.
4. SI Table S1: keep the old \(R^2\) comparison only as a diagnostic or replace it with the AIC/BIC compact table from `{COMPACT_CSV}`.
5. Response letter: lead with the likelihood comparison and say plainly that the revision softened the original power-law framing.

## If Reviewer 2 attacks the messy result

The fallback is strong and honest: “Our central conclusion does not rely on proving pure power-law statistics. The model comparison was added specifically to avoid overinterpreting log-log plots. The revised conclusion is that solvation-shell kinetics are non-Poissonian and temporally heterogeneous, with finite-window heavy-tailed regimes, and that the event-based two-state classification links this heterogeneity to physically interpretable soft and hard solvation environments.”
"""


def replace_section(text: str, start: str, end: str, replacement: str) -> str:
    i = text.index(start)
    j = text.index(end, i)
    return text[:i] + replacement + text[j:]


def update_response_draft() -> None:
    path = RESP / 'response_to_reviewers_draft.md'
    text = path.read_text()
    start = '## Reviewer 2 Comment 1: Evidence for power-law-like behavior\n'
    end = '## Reviewer 2 Comment 2: Threshold dependence and algorithm dependence\n'
    new_section = r"""## Reviewer 2 Comment 1: Evidence for power-law-like behavior

One of the main concerns is the evidence for power-law-like behavior. The log-log plots are suggestive, but visual linearity over a limited range is not sufficient to establish power-law or heavy-tailed dynamics. The authors should compare alternative models, such as stretched exponentials, truncated power laws, exponential cutoffs, or mixtures of relaxation times. The use of R2 alone is not enough. A more rigorous statistical comparison using likelihood-based fitting, AIC/BIC, or goodness-of-fit tests would make the conclusions much stronger.

Author reply

Current response direction:
- Agree explicitly that log-log visual linearity and R2 are insufficient.
- Add likelihood-based model comparison over common fitting windows.
- Compare conditional exponential, Weibull/stretched-exponential, Pareto power-law, tempered power-law, and biexponential mixture models.
- Report log-likelihood and AIC/BIC in SI, with a compact summary in the main text.
- State the important negative result honestly: pure power-law models do not uniquely dominate the full pair-survival distributions.
- Reframe the conclusion as non-single-exponential, temporally heterogeneous, finite-window tail behavior rather than asymptotic power-law statistics.

Draft response text:

We agree with the reviewer that log-log visual linearity and R2 comparisons are insufficient to establish a power-law distribution. In the revised manuscript, we therefore added a likelihood-based model comparison over common fitting windows. Specifically, we compared conditional exponential, conditional Weibull/stretched-exponential, Pareto power-law, tempered power-law, and conditional biexponential-mixture models using log-likelihood, AIC, and BIC. This analysis showed that the simple single-exponential reference is strongly disfavored for most state-resolved pair-survival and shell-change inter-event distributions. At the same time, the analysis also showed that pure power-law models do not uniquely dominate the full pair-survival distributions. We therefore revised the language throughout the manuscript to describe the fitted exponents as finite-window tail-heaviness descriptors, rather than asymptotic universal power-law exponents, and we added the model-comparison results to the Supporting Information.

Result to cite in response:
- Full state-resolved pair survival: best AIC counts are biexponential mixture 13/24, Weibull/stretched exponential 9/24, and tempered power law 2/24.
- Pair-survival upper-quartile tails: best AIC counts are biexponential mixture 14/23, conditional exponential 4/23, Weibull/stretched exponential 3/23, and tempered power law 2/23.
- Shell-change inter-event upper-quartile tails: tempered power law is best in 6/12 systems, biexponential mixture in 5/12, and Weibull/stretched exponential in 1/12.
- The single-exponential reference is usually strongly penalized, supporting non-single-exponential temporal heterogeneity even when pure power-law language is softened.

Manuscript changes:
- Planned: add model-comparison paragraph to the main text near the beta/statistical-caution discussion.
- Planned: add or replace SI table with likelihood/AIC/BIC model-comparison table.
- Planned: make terminology pass replacing strong “power-law” claims with “finite-window power-law-like”, “tail-heaviness descriptor”, or “non-single-exponential temporal heterogeneity”.

Page number:
- TBD after manuscript edits.

Checklist link:
- R2-1 in `/nas_2/transcendence/revision/notes/00_project_management/checklist.md`

Generated outputs:
- `/nas_2/transcendence/revision/analysis/powerlaw_model_comparison/outputs/model_comparison_all.csv`
- `/nas_2/transcendence/revision/notes/01_r2_1_distribution_statistics/analysis_output_summaries/model_comparison_summary.md`
- `/nas_2/transcendence/revision/analysis/powerlaw_model_comparison/outputs/r2_1_best_model_compact_for_si.csv`
- `/nas_2/transcendence/revision/notes/01_r2_1_distribution_statistics/r2_1_powerlaw_rescue_pack.md`

"""
    path.write_text(replace_section(text, start, end, new_section))


def update_checklist() -> None:
    path = REV / 'notes' / '00_project_management' / 'checklist.md'
    text = path.read_text()
    old = '| R2-1 | 어느정도 구현 | Provide stronger evidence for power-law-like/heavy-tailed behavior; log-log plots and R2 are insufficient. Compare alternative models and preferably use likelihood/AIC/BIC/goodness-of-fit tests. | Manuscript uses cautious language and cites power-law statistics papers, but quantitative model comparison is weak. | Fit/compare exponential, stretched exponential, finite-window power law, and tempered/truncated power law on common windows. Report log-likelihood, AIC/BIC, and short conclusion. | Main claim becomes "heavier-tailed than single exponential over fitted finite windows", not "proved pure power law". | `analysis/powerlaw_model_comparison/`, `notes/01_r2_1_distribution_statistics/powerlaw_model_comparison_interpretation.md`, candidate SI table |'
    new = '| R2-1 | 어느정도 구현 | Provide stronger evidence for power-law-like/heavy-tailed behavior; log-log plots and R2 are insufficient. Compare alternative models and preferably use likelihood/AIC/BIC/goodness-of-fit tests. | First-pass likelihood/AIC/BIC comparison is complete. Result: pure power law does not uniquely dominate full pair-survival distributions, but single exponential is strongly disfavored and shell-change inter-event tails often favor tempered power law. | Insert compact AIC/BIC table into SI, add a short main-text paragraph, and do a terminology pass that softens strong power-law claims. | Main claim becomes "non-single-exponential, temporally heterogeneous finite-window tail behavior", not "proved pure power law". | `analysis/powerlaw_model_comparison/`, `outputs/r2_1_best_model_compact_for_si.csv`, `notes/01_r2_1_distribution_statistics/r2_1_powerlaw_rescue_pack.md` |'
    if old in text:
        path.write_text(text.replace(old, new))
        return
    if new in text:
        return
    raise RuntimeError('R2-1 checklist row not found; refusing blind edit')


def append_change_log() -> None:
    path = REV / 'notes' / '00_project_management' / 'change_log.md'
    text = path.read_text()
    entry = '- Built R2-1 response/rescue pack and SI-ready compact model-comparison table; updated response draft and checklist with the softened power-law strategy.\n'
    if entry not in text:
        path.write_text(text.rstrip() + '\n' + entry)


def main() -> None:
    all_rows = read_rows(ALL_CSV)
    count_rows = read_rows(COUNTS_CSV)
    compact = best_rows(all_rows)
    write_compact_csv(compact)
    PACK_MD.write_text(make_pack(count_rows, compact))
    update_response_draft()
    update_checklist()
    append_change_log()
    print(f'wrote {COMPACT_CSV}')
    print(f'wrote {PACK_MD}')
    print('updated response draft, checklist, and change log')


if __name__ == '__main__':
    main()
