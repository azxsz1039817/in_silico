from __future__ import annotations

import argparse
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils import add_common_args


DOMAIN_GROUPS = {
    "vaccine": {
        "title": "Vaccine justifications",
        "domains": ["vaccine_get", "vaccine_not_get"],
        "output": "figures/justification_vaccine_bars.png",
    },
    "mask_mandate": {
        "title": "Mask mandate justifications",
        "domains": ["mask_mandate_for", "mask_mandate_against"],
        "output": "figures/justification_mask_mandate_bars.png",
    },
    "fear_virus": {
        "title": "Fear of COVID-19 justifications",
        "domains": ["fear_virus_for", "fear_virus_against"],
        "output": "figures/justification_fear_virus_bars.png",
    },
    "appendix": {
        "title": "Appendix mask-mandate justification checks",
        "domains": [
            "appendix_mask_mandate_for_alt1",
            "appendix_mask_mandate_against_alt",
            "appendix_mask_mandate_for_alt2",
        ],
        "output": "figures/justification_appendix_bars.png",
    },
}

DOMAIN_LABELS = {
    "vaccine_get": "For getting vaccine",
    "vaccine_not_get": "Against getting vaccine",
    "mask_mandate_for": "For mask mandate",
    "mask_mandate_against": "Against mask mandate",
    "fear_virus_for": "Afraid of virus",
    "fear_virus_against": "Not afraid of virus",
    "appendix_mask_mandate_for_alt1": "Appendix: for mask mandate 1",
    "appendix_mask_mandate_against_alt": "Appendix: against mask mandate",
    "appendix_mask_mandate_for_alt2": "Appendix: for mask mandate 2",
}

CATEGORY_LABELS = {
    "personal_social_responsibility": "Personal/\nsocial resp.",
    "trust_science_government": "Trust in\nscience/gov.",
    "risk_weighing": "Weighing\nrisks/benefits",
    "fear_disease": "Fear of\ndisease impact",
    "distrust_government_pharma": "Distrust gov./\npharma",
    "freedom_skepticism": "Freedom/\nskepticism",
    "health_side_effects": "Health/side\neffects",
    "virus_transmission": "Virus\ntransmission",
    "political_identity": "Political\nidentity",
    "public_health": "Public\nhealth",
    "government_responsibility": "Government\nresponsibility",
    "efficacy_misconceptions": "Efficacy\nmisconceptions",
    "government_mandates": "Government\nmandates",
    "individual_freedom": "Individual\nfreedom",
    "mortality_contagiousness": "Mortality/\ncontagiousness",
    "global_threat": "Global\nthreat",
    "scientific_consensus": "Scientific\nconsensus",
    "government_distrust": "Government\ndistrust",
    "personal_experience": "Personal\nexperience",
    "opinions_response": "Opinions on\nresponse",
    "impacts_comparisons": "Impacts and\ncomparisons",
    "low_perceived_threat": "Low perceived\nthreat",
    "threat_minimization": "Threat\nminimization",
    "trust_science_government": "Trust in\nscience/gov.",
    "public_health_economic": "Public health/\neconomic",
    "government_responsibility_rights": "Government resp./\nrights",
    "pro_mask_perspectives": "Pro-mask\nperspectives",
    "government_overreach_freedom": "Gov. overreach/\nfreedom",
    "concerns_mask_mandates": "Concerns over\nmask mandates",
    "government_responsibility_rights": "Government resp./\nrights",
    "political_perspectives": "Political\nperspectives",
    "pro_mask_public_safety": "Public safety\nconcerns",
    "science_effectiveness": "Science/\neffectiveness",
}

COLORS = {"liberal": "#5b9bd5", "conservative": "#d95b54"}


def category_label(value: str) -> str:
    if value in CATEGORY_LABELS:
        return CATEGORY_LABELS[value]
    text = value.replace("_", " ").title()
    return "\n".join(textwrap.wrap(text, width=16, break_long_words=False))


def table_to_bars(tests: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for row in tests.itertuples(index=False):
        right, left = str(row.comparison).split("_vs_", maxsplit=1)
        candidates = [
            (right, int(row.lib_right), int(row.con_right), float(row.p_value_greater)),
            (left, int(row.lib_left), int(row.con_left), None),
        ]
        for category, liberal, conservative, p_value in candidates:
            key = (str(row.domain), category)
            if key in seen:
                continue
            seen.add(key)
            total = liberal + conservative
            rows.append(
                {
                    "domain": row.domain,
                    "category": category,
                    "liberal": liberal,
                    "conservative": conservative,
                    "total": total,
                    "liberal_share": liberal / total if total else 0,
                    "p_value_greater": p_value,
                }
            )
    return pd.DataFrame(rows)


def plot_domain_group(bars: pd.DataFrame, group_key: str, output_path) -> None:
    spec = DOMAIN_GROUPS[group_key]
    domains = [d for d in spec["domains"] if d in set(bars["domain"])]
    if not domains:
        return

    max_total = float(bars[bars["domain"].isin(domains)]["total"].max())
    fig, axes = plt.subplots(1, len(domains), figsize=(max(7.5, 4.0 * len(domains)), 4.2), squeeze=False, sharey=True)
    for ax, domain in zip(axes.ravel(), domains):
        sub = bars[bars["domain"].eq(domain)].copy()
        x = range(len(sub))
        ax.bar(x, sub["liberal"], color=COLORS["liberal"], edgecolor="#6f6f6f", linewidth=0.7, label="Liberal")
        ax.bar(
            x,
            sub["conservative"],
            bottom=sub["liberal"],
            color=COLORS["conservative"],
            edgecolor="#6f6f6f",
            linewidth=0.7,
            label="Conservative",
        )
        for idx, row in enumerate(sub.itertuples(index=False)):
            if row.liberal:
                ax.text(idx, row.liberal / 2, f"{row.liberal}\n({row.liberal_share:.1%})", ha="center", va="center", fontsize=8)
            if row.conservative:
                con_share = row.conservative / row.total
                ax.text(idx, row.liberal + row.conservative / 2, f"{row.conservative}\n({con_share:.1%})", ha="center", va="center", fontsize=8)
            if pd.notna(row.p_value_greater) and row.p_value_greater < 0.05:
                ax.text(idx, row.total + max(sub["total"]) * 0.035, "*", ha="center", va="bottom", fontsize=15, fontweight="bold")
        ax.set_title(DOMAIN_LABELS.get(domain, domain.replace("_", " ").title()), fontsize=12, fontweight="bold")
        ax.set_xticks(list(x))
        ax.set_xticklabels([category_label(v) for v in sub["category"]], rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, max_total * 1.14)
        ax.grid(axis="y", color="#eeeeee", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    axes.ravel()[0].set_ylabel("Frequency by partisan classification")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    fig.suptitle(spec["title"], fontsize=15, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0.07, 1, 0.91))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Generate manuscript-style justification bar charts."))
    parser.add_argument("--tests", default="tables/justification_two_proportion_tests.csv", help="Two-proportion test table under project root.")
    parser.add_argument("--group", choices=sorted(DOMAIN_GROUPS), default=None, help="Generate only one domain group.")
    args = parser.parse_args()

    tests_path = args.project_root / args.tests
    tests = pd.read_csv(tests_path)
    bars = table_to_bars(tests)
    bars.to_csv(args.project_root / "tables" / "justification_bar_counts.csv", index=False)

    groups = [args.group] if args.group else list(DOMAIN_GROUPS)
    for group_key in groups:
        output_path = args.project_root / DOMAIN_GROUPS[group_key]["output"]
        plot_domain_group(bars, group_key, output_path)


if __name__ == "__main__":
    main()
