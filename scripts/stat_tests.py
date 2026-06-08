from __future__ import annotations

import argparse
import math

import pandas as pd

from utils import add_common_args, exact_binom_greater, write_csv, write_json


JUSTIFICATION_TESTS = [
    ("vaccine_get", "personal_social_responsibility_vs_trust_science_government", 172, 121, 226, 235),
    ("vaccine_get", "risk_weighing_vs_trust_science_government", 335, 207, 226, 235),
    ("vaccine_get", "fear_disease_vs_trust_science_government", 230, 120, 226, 235),
    ("vaccine_not_get", "distrust_government_pharma_vs_freedom_skepticism", 114, 217, 89, 201),
    ("vaccine_not_get", "health_side_effects_vs_freedom_skepticism", 169, 253, 89, 201),
    ("mask_mandate_for", "virus_transmission_vs_political_identity", 241, 164, 222, 157),
    ("mask_mandate_for", "public_health_vs_political_identity", 323, 180, 222, 157),
    ("mask_mandate_for", "government_responsibility_vs_political_identity", 264, 124, 222, 157),
    ("mask_mandate_against", "efficacy_misconceptions_vs_government_mandates", 187, 312, 96, 296),
    ("mask_mandate_against", "individual_freedom_vs_government_mandates", 149, 243, 96, 296),
    ("fear_virus_for", "mortality_contagiousness_vs_global_threat", 199, 235, 137, 232),
    ("fear_virus_for", "scientific_consensus_vs_global_threat", 146, 172, 137, 232),
    ("fear_virus_for", "government_distrust_vs_global_threat", 179, 207, 137, 232),
    ("fear_virus_for", "personal_experience_vs_global_threat", 253, 214, 137, 232),
    ("fear_virus_against", "opinions_response_vs_impacts_comparisons", 61, 49, 78, 84),
    ("fear_virus_against", "low_perceived_threat_vs_impacts_comparisons", 182, 143, 78, 84),
    ("fear_virus_against", "threat_minimization_vs_impacts_comparisons", 115, 88, 78, 84),
    ("fear_virus_against", "trust_science_government_vs_impacts_comparisons", 136, 60, 78, 84),
    ("appendix_mask_mandate_for_alt1", "public_health_economic_vs_political_identity", 239, 152, 250, 177),
    ("appendix_mask_mandate_for_alt1", "government_responsibility_rights_vs_political_identity", 275, 150, 250, 177),
    ("appendix_mask_mandate_for_alt1", "pro_mask_perspectives_vs_political_identity", 275, 137, 250, 177),
    ("appendix_mask_mandate_against_alt", "political_identity_vs_government_overreach_freedom", 142, 284, 111, 267),
    ("appendix_mask_mandate_against_alt", "concerns_mask_mandates_vs_government_overreach_freedom", 180, 300, 111, 267),
    ("appendix_mask_mandate_for_alt2", "government_responsibility_rights_vs_political_perspectives", 251, 157, 253, 192),
    ("appendix_mask_mandate_for_alt2", "pro_mask_public_safety_vs_political_perspectives", 237, 131, 253, 192),
    ("appendix_mask_mandate_for_alt2", "science_effectiveness_vs_political_perspectives", 286, 127, 253, 192),
]


def two_proportion_z_greater(success_right: int, fail_right: int, success_left: int, fail_left: int) -> dict[str, float]:
    n_right = success_right + fail_right
    n_left = success_left + fail_left
    p_right = success_right / n_right
    p_left = success_left / n_left
    pooled = (success_right + success_left) / (n_right + n_left)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_right + 1 / n_left))
    z = (p_right - p_left) / se
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return {"p_right": p_right, "p_left": p_left, "difference": p_right - p_left, "z": z, "p_value_greater": p}


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Run manuscript binomial and justification proportion tests."))
    args = parser.parse_args()
    table_dir = args.project_root / "tables"

    binomial = pd.DataFrame(
        [
            {"level": "prompt", "null_handling": "count_null_as_wrong", "correct": 109, "total": 179},
            {"level": "prompt", "null_handling": "exclude_null", "correct": 109, "total": 121},
            {"level": "issue", "null_handling": "count_null_as_wrong", "correct": 37, "total": 49},
            {"level": "issue", "null_handling": "exclude_null", "correct": 37, "total": 44},
        ]
    )
    binomial["p_value_greater"] = [exact_binom_greater(int(r.correct), int(r.total)) for r in binomial.itertuples()]
    write_csv(binomial, table_dir / "forecast_binomial_tests.csv")

    rows = []
    for domain, comparison, lib_right, con_right, lib_left, con_left in JUSTIFICATION_TESTS:
        result = two_proportion_z_greater(lib_right, con_right, lib_left, con_left)
        rows.append(
            {
                "domain": domain,
                "comparison": comparison,
                "lib_right": lib_right,
                "con_right": con_right,
                "lib_left": lib_left,
                "con_left": con_left,
                **result,
            }
        )
    just = pd.DataFrame(rows)
    write_csv(just, table_dir / "justification_two_proportion_tests.csv")
    write_json({"binomial_tests": len(binomial), "justification_tests": len(just)}, table_dir / "stat_tests_summary.json")


if __name__ == "__main__":
    main()
