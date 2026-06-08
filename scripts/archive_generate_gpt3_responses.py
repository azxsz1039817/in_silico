from __future__ import annotations

import argparse
import os
import time

import pandas as pd

from utils import add_common_args, prompt_bank_path, write_csv


COVID_INTROS = [
    "I am a strong liberal and a lifelong Democrat. In 2016, I was proud to vote for Hillary Clinton and I think that the Republicans have been a disaster for this country. Lately, one of the biggest political issues has been the COVID-19 pandemic caused by the new coronavirus.",
    "I am a strong conservative and a lifelong Republican. In 2016, I was proud to vote for Donald Trump and I think that the Democrats have been a disaster for this country. Lately, one of the biggest political issues has been the COVID-19 pandemic caused by the new coronavirus.",
]

VALIDATION_INTROS = [
    "I am a strong liberal and a lifelong Democrat. In 2016, I was proud to vote for Hillary Clinton and I think that the Republicans have been a disaster for this country.",
    "I am a strong conservative and a lifelong Republican. In 2016, I was proud to vote for Donald Trump and I think that the Democrats have been a disaster for this country.",
]

ENDING_TEMPLATES = [
    " My stance on {} is that it is a",
    " In my opinion, {} is a",
    " I think {} is a",
    " I believe that {} is a",
    " My personal opinion on {} is that it is a",
    " I personally believe {} is a",
]


def build_prompts(dataset: str, root) -> pd.DataFrame:
    prompt_dataset = "generation_covid" if dataset == "covid" else "validation"
    table = pd.read_csv(prompt_bank_path(prompt_dataset, root), encoding="utf-8-sig")
    intros = COVID_INTROS if dataset == "covid" else VALIDATION_INTROS
    records = []
    prompt_no = -1
    question_no = -1
    for row in table.itertuples(index=False):
        endings = [int(x) for x in str(row.endings).split(",") if str(x).strip()]
        keywords = [x.strip() for x in str(row.keywords).split(",") if x.strip()]
        for ending_no in endings:
            template = ENDING_TEMPLATES[ending_no]
            for keyword in keywords:
                question_no += 1
                for partisanship, intro in enumerate(intros):
                    prompt_no += 1
                    ending = template.format(keyword)
                    records.append(
                        {
                            "prompt": f"{intro} {row.prompt}{ending}",
                            "issue": row.prompt,
                            "partisanship": partisanship,
                            "ending": ending,
                            "keyword": keyword,
                            "question_no": question_no,
                            "prompt_no": prompt_no,
                            "issue_no": row.issue_no,
                            "labels": row.labels,
                        }
                    )
    return pd.DataFrame(records)


def run_legacy_completion(prompt: str, *, model: str, n: int, max_tokens: int, temperature: float) -> list[str]:
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
        n=n,
    )
    return [choice.text.strip() for choice in response.choices]


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Historical GPT-3 completion generation script."))
    parser.add_argument("--dataset", choices=["covid", "validation"], required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model", default="davinci")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dry-run", action="store_true", help="Only write the generated prompt table; do not call OpenAI.")
    args = parser.parse_args()

    prompts = build_prompts(args.dataset, args.project_root)
    output_dir = args.project_root / (args.output_dir or f"data/raw/archive/archive_{args.dataset}_rerun")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(prompts, output_dir / "all_prompts.csv")

    if args.dry_run:
        return
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set for historical API reruns.")

    for row in prompts.itertuples(index=False):
        completions = run_legacy_completion(row.prompt, model=args.model, n=args.n, max_tokens=args.max_tokens, temperature=args.temperature)
        out = pd.DataFrame({"completion": completions, "prompt_no": row.prompt_no, "prompt": row.prompt, "question_no": row.question_no})
        write_csv(out, output_dir / f"completions_prompt_{row.prompt_no}.csv")
        time.sleep(0.2)


if __name__ == "__main__":
    main()
