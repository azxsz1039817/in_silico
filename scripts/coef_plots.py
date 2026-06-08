from __future__ import annotations

import argparse
import math
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from assemble_responses import assemble
from utils import add_common_args, slope_and_se, write_csv


COVID_GROUPS = {
    "vaccine": list(range(0, 10)) + [20, 22, 25, 26, 28, 31, 32],
    "masks": [17, 18, 19, 21, 27, 29, 30, 36, 37],
    "lockdown_travel": [10, 11, 12, 13, 14, 15, 16, 23, 24],
    "cdc_virus_misc": [33, 34, 35, 38, 39],
}

VALIDATION_GROUPS = {
    "part1": list(range(0, 12)),
    "part2": list(range(12, 28)),
}

COVID_GROUP_LABELS = {
    "vaccine": "COVID: vaccine items",
    "masks": "COVID: mask items",
    "lockdown_travel": "COVID: lockdown and travel items",
    "cdc_virus_misc": "COVID: CDC, virus-origin, and miscellaneous items",
}

PANEL_LABELS = {
    0.1: "Requiring government workers to get the vaccine",
    1.1: "Allowing government workers to return without getting the vaccine",
    2.1: "Requiring students to get the vaccine",
    3.1: "Allowing students to return without getting the vaccine",
    4.1: "Requiring proof of vaccination to travel by plane",
    5.1: "Requiring proof of vaccination to enter bars or restaurants",
    6.1: "Requiring proof of vaccination to attend large public gatherings",
    7.1: "Allowing unvaccinated people to enter bars and restaurants",
    8.1: "Allowing unvaccinated people to travel by plane",
    9.1: "Opening schools without a vaccine requirement",
    10.1: "Closing businesses until infection rates go down",
    11.1: "Closing bars and restaurants until infection rates go down",
    12.1: "Prohibiting large gatherings until infection rates go down",
    13.1: "Avoiding small gatherings until infection rates go down",
    14.1: "Closing businesses",
    14.2: "Keeping businesses open",
    15.1: "Keeping schools open",
    15.2: "Closing schools",
    16.1: "Closing schools and conducting classes online",
    16.2: "Switching to remote schooling",
    17.1: "A mask mandate",
    17.2: "A mask requirement",
    18.1: "Mandating masks in stores and workplaces",
    18.2: "Requiring masks in stores and workplaces",
    19.1: "Mandating masks in schools",
    19.2: "Requiring masks in schools",
    20.1: "A vaccine mandate",
    20.2: "A vaccine requirement",
    21.1: "Ending mask mandates",
    22.1: "Ending vaccine mandates",
    23.1: "Banning visitors from countries with COVID-19 outbreaks",
    24.1: "Stopping international travel",
    25.1: "Letting government workers opt out of the vaccine",
    26.1: "Letting students opt out of the vaccine",
    27.1: "Letting individuals choose whether or not to mask",
    27.2: "Letting people decide whether or not to wear a mask",
    28.1: "Letting individuals choose whether or not to get the vaccine",
    28.2: "Letting people decide whether or not to get vaccinated",
    29.1: "Whether to wear a mask to slow the spread of the virus",
    30.1: "Whether to wear a mask to protect yourself from the virus",
    31.1: "Whether to get the vaccine to slow the spread of the virus",
    32.1: "Whether to get the vaccine to protect yourself from the virus",
    33.1: "Whether the CDC is doing a good job handling the situation",
    34.1: "Whether the CDC is exaggerating the danger posed by the virus",
    35.1: "Whether the virus is something we should be afraid of",
    36.1: "Whether face masks should be optional or mandatory",
    37.1: "Whether surgical masks slow the spread of the virus",
    38.1: "Whether the virus originated in a wild animal or a laboratory",
    39.1: "Whether the virus originated in a wild animal or a laboratory in China",
}

VALIDATION_PANEL_LABELS = {
    0.1: "Banning abortion",
    0.2: "Outlawing abortion",
    1.1: "Increasing access to abortion",
    1.2: "Making it easier to get an abortion",
    2.1: "Pro-life or pro-choice",
    3.1: "Passing climate legislation",
    3.2: "Taking action against climate change",
    4.1: "Switching to renewables",
    4.2: "Transitioning away from fossil fuels",
    5.1: "Global warming is a threat or not",
    6.1: "Outlawing gay marriage",
    6.2: "Preserving gay marriage",
    7.1: "Allowing transgender people in the military",
    7.2: "Barring transgender people from the military",
    8.1: "Trying to improve gender equality",
    9.1: "Legalizing marijuana",
    10.1: "Decreasing police budgets",
    10.2: "Law enforcement reform",
    11.1: "Increasing policing",
    11.2: "Strengthening the police",
    12.1: "Affirmative action",
    13.1: "Racism is a major cause of inequality in the U.S.",
    14.1: "Increasing immigration",
    15.1: "Cracking down on illegal immigration",
    15.2: "Heightening immigration enforcement",
    16.1: "Increasing gun control",
    17.1: "Protecting gun rights",
    18.1: "Universal healthcare",
    19.1: "Subsidized healthcare",
    20.1: "Taxing the wealthy",
    21.1: "Cutting corporate taxes",
    22.1: "Increasing welfare",
    23.1: "Cutting social programs",
    24.1: "Decreasing regulation",
    25.1: "Regulating corporations",
    26.1: "Increasing the minimum wage",
    27.1: "Unions",
}

COLORS = {"liberal": "#2f73b7", "conservative": "#c44e52", "no_effect": "#9a9a9a"}


def wrap_label(value: object, width: int = 22) -> str:
    text = "" if pd.isna(value) else str(value)
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def issue_label(row: pd.Series, panel_labels: dict[float, str]) -> str:
    key = float(row["issue_no_key"])
    return panel_labels.get(key, str(row.get("keyword", key)))


def axis_label(row: pd.Series) -> str:
    return f"{row.get('label_1', '')} -\n{row.get('label_2', '')}"


def coefficient_table(df: pd.DataFrame) -> pd.DataFrame:
    named = df[df["named_prime"].eq(1)].copy()
    rows: list[dict[str, object]] = []
    for (issue_no_key, question_no), group in named.groupby(["issue_no_key", "question_no"], sort=True):
        coef, se, n = slope_and_se(group["liberal"], group["proj"])
        first = group.iloc[0]
        labels = str(first.get("labels", "")).split(",")
        rows.append(
            {
                "issue_no_key": issue_no_key,
                "question_no": question_no,
                "issue_no": first.get("issue_no"),
                "keyword": first.get("keyword"),
                "ending": first.get("ending"),
                "end_type": first.get("end_type"),
                "label_1": labels[0].strip() if labels else "",
                "label_2": labels[1].strip() if len(labels) > 1 else "",
                "coef": coef,
                "se": se,
                "n": n,
                "ci_low": coef - 1.98 * se,
                "ci_high": coef + 1.98 * se,
                "direction": "liberal" if coef - 1.98 * se > 0 else "conservative" if coef + 1.98 * se < 0 else "no_effect",
            }
        )
    return pd.DataFrame(rows)


def plot_coefficients(coefs: pd.DataFrame, output_path, title: str, panel_labels: dict[float, str]) -> None:
    if coefs.empty:
        return
    plot_df = coefs.sort_values(["issue_no", "issue_no_key", "question_no"]).reset_index(drop=True)
    panels = list(plot_df["issue_no_key"].drop_duplicates())
    n_cols = min(4, max(1, len(panels)))
    n_rows = math.ceil(len(panels) / n_cols)
    fig_w = 3.2 * n_cols
    fig_h = 2.5 * n_rows + 0.8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False, sharey=True)

    y_min = min(-0.08, float(plot_df["ci_low"].min()) - 0.01)
    y_max = max(0.08, float(plot_df["ci_high"].max()) + 0.01)
    label_pairs = plot_df[["label_1", "label_2"]].drop_duplicates()
    mixed_axes = len(label_pairs) > 1
    for ax, issue_no_key in zip(axes.ravel(), panels):
        sub = plot_df[plot_df["issue_no_key"].eq(issue_no_key)].copy()
        x = list(range(len(sub)))
        colors = sub["direction"].map(COLORS)
        coef = sub["coef"].to_numpy()
        err_low = (sub["coef"] - sub["ci_low"]).to_numpy()
        err_high = (sub["ci_high"] - sub["coef"]).to_numpy()
        ax.errorbar(
            x,
            coef,
            yerr=[err_low, err_high],
            fmt="none",
            ecolor="#8f8f8f",
            alpha=0.65,
            linewidth=1,
            capsize=3,
        )
        ax.scatter(x, coef, c=colors, s=30, alpha=0.9, zorder=3)
        ax.axhline(0, color="#222222", linestyle="--", linewidth=0.9)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(list(x))
        ax.set_xticklabels([wrap_label(v, 14) for v in sub["end_type"]], rotation=35, ha="right", fontsize=7)
        ax.set_title(wrap_label(issue_label(sub.iloc[0], panel_labels), 24), fontsize=9, pad=6)
        if mixed_axes:
            ax.set_ylabel(axis_label(sub.iloc[0]), fontsize=7, color="#555555", labelpad=7)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.ravel()[len(panels) :]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.supxlabel("Prompt wording", fontsize=10)
    if mixed_axes:
        fig.supylabel("OLS coefficient for liberal prime", fontsize=10)
    else:
        label_1 = str(plot_df.iloc[0].get("label_1", "label 1"))
        label_2 = str(plot_df.iloc[0].get("label_2", "label 2"))
        fig.supylabel(f"OLS coefficient for liberal prime\n({label_1} - {label_2})", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Generate coefficient tables and simple manuscript-style coefficient plots."))
    parser.add_argument("--dataset", choices=["covid", "validation"], required=True)
    args = parser.parse_args()

    processed_path = args.project_root / "data" / "processed" / f"{args.dataset}_responses.csv"
    if processed_path.exists():
        responses = pd.read_csv(processed_path)
    else:
        responses = assemble(args.dataset, args.project_root)

    coefs = coefficient_table(responses)
    table_dir = args.project_root / "tables"
    fig_dir = args.project_root / "figures"
    write_csv(coefs, table_dir / f"{args.dataset}_coefficients.csv")
    if args.dataset == "covid":
        for name, issue_numbers in COVID_GROUPS.items():
            sub = coefs[coefs["issue_no"].isin(issue_numbers)]
            write_csv(sub, table_dir / f"covid_{name}_coefficients.csv")
            plot_coefficients(sub, fig_dir / f"covid_{name}_coefficients.png", COVID_GROUP_LABELS[name], PANEL_LABELS)
        stale_summary = fig_dir / "covid_coefficients.png"
        if stale_summary.exists():
            stale_summary.unlink()
    else:
        for name, issue_numbers in VALIDATION_GROUPS.items():
            sub = coefs[coefs["issue_no"].isin(issue_numbers)]
            plot_coefficients(
                sub,
                fig_dir / f"validation_coefficients_{name}.png",
                f"Validation coefficient estimates: {name.replace('part', 'part ')}",
                VALIDATION_PANEL_LABELS,
            )
        for stale_name in ("validation_coefficients.png", "validation_coefficients_summary.png"):
            stale_path = fig_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()


if __name__ == "__main__":
    main()
