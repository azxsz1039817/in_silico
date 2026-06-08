from __future__ import annotations

import argparse
import os
import time

import pandas as pd

from utils import add_common_args, write_csv


def run_legacy_justification(prompt: str, *, model: str, max_tokens: int, temperature: float) -> str:
    import openai

    openai.api_key = os.environ["OPENAI_API_KEY"]
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
        frequency_penalty=1.2,
        presence_penalty=0.0,
    )
    return response.choices[0].text.strip()


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Historical GPT-3 justification generation from cached completions."))
    parser.add_argument("--responses", default="data/processed/covid_responses_named.csv")
    parser.add_argument("--output", default="data/processed/archival_justification_prompts.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="davinci")
    parser.add_argument("--max-tokens", type=int, default=70)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dry-run", action="store_true", help="Write justification prompts only; do not call OpenAI.")
    args = parser.parse_args()

    responses = pd.read_csv(args.project_root / args.responses)
    prompts = responses.assign(justification_prompt=lambda d: d["full_prompt"].astype(str) + " " + d["completion"].astype(str) + ". This is because")
    if args.limit:
        prompts = prompts.head(args.limit)
    write_csv(prompts[["prompt_no", "completion_no", "justification_prompt"]], args.project_root / args.output)
    if args.dry_run:
        return
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set for historical API reruns.")
    rows = []
    for row in prompts.itertuples(index=False):
        rows.append(
            {
                "prompt_no": row.prompt_no,
                "completion_no": row.completion_no,
                "justification": run_legacy_justification(row.justification_prompt, model=args.model, max_tokens=args.max_tokens, temperature=args.temperature),
            }
        )
        time.sleep(0.2)
    write_csv(pd.DataFrame(rows), args.project_root / "data" / "processed" / "archival_justifications_rerun.csv")


if __name__ == "__main__":
    main()

