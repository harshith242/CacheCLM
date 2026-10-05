"""CLI: `cacheclm smoke` (1 sample, 5 questions, 3 arms), `cacheclm run` (the 7 samples x 3 arms x 2 repeats),
`cacheclm report [--smoke]`. Settings in configs/base.yaml; DEEPSEEK_API_KEY from a .env in this or a parent folder."""
import argparse
import os

import openai
import yaml
from dotenv import find_dotenv, load_dotenv

from cacheclm.arms import ARMS
from cacheclm.budget import Budget, BudgetExceeded
from cacheclm.llm import LLM
from cacheclm.mab import load_sample
from cacheclm.report import write_report
from cacheclm.run import run_sample


def make_llm(cfg, price, budget):
    agent = cfg["agent"]
    client = openai.OpenAI(base_url=agent["base_url"], api_key=os.environ[agent["api_key_env"]], max_retries=0,
                           timeout=300)
    return LLM(agent["model"], client, cfg["cache_dir"], cfg["temperature"], price, agent.get("options"), budget.spend)


def main():
    parser = argparse.ArgumentParser(prog="cacheclm")
    parser.add_argument("command", choices=["smoke", "run", "report"])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--smoke", action="store_true", help="report on the smoke run")
    args = parser.parse_args()
    load_dotenv(find_dotenv(usecwd=True))
    cfg = yaml.safe_load(open(args.config))
    prices = yaml.safe_load(open(cfg["prices"]))
    smoke = args.command == "smoke" or args.smoke
    runs_dir = cfg["smoke"]["runs_dir"] if smoke else cfg["runs_dir"]
    if args.command == "report":
        write_report(runs_dir, cfg["smoke"]["results_dir"] if smoke else cfg["results_dir"], prices)
        return
    budget = Budget(cfg["budget_file"], cfg["budget_usd"])
    llm = make_llm(cfg, prices["deepseek"], budget)
    samples = [cfg["smoke"]["sample"]] if smoke else cfg["samples"]
    repeats = [0] if smoke else cfg["repeats"]
    limit = cfg["smoke"]["questions"] if smoke else None
    try:
        for repeat in repeats:
            for spec in samples:
                sample = load_sample(cfg["data_dir"], spec["split"], spec["row"])
                for arm in ARMS:
                    try:
                        acc = run_sample(sample, arm, repeat, llm, cfg, prices["deepseek"], runs_dir, limit)
                    except RuntimeError as e:  # API kept failing after retries: skip, the report lists it as incomplete
                        print(f"FAILED   {arm:8s} {sample.sid:24s} r{repeat}: {e}")
                        continue
                    print(f"{arm:8s} {sample.sid:24s} r{repeat}  accuracy {acc:.2f}  total spend ${budget.total:.3f}")
    except BudgetExceeded as e:
        print(f"Stopped: {e}. Finished runs are kept; raise budget_usd in the config to continue.")


if __name__ == "__main__":
    main()
