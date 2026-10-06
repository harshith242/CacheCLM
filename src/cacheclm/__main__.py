"""CLI: `cacheclm smoke` (the smoke samples, 4 arms), `cacheclm run` (the samples, 4 arms plus the reference arms, and
the extra FactConsolidation repeat), `cacheclm report [--smoke]`. Settings in configs/base.yaml; DEEPSEEK_API_KEY from .env."""
import argparse
import os

import openai
import yaml
from dotenv import find_dotenv, load_dotenv

from cacheclm.arms import ARMS, REFERENCES
from cacheclm.budget import Budget, BudgetExceeded
from cacheclm.llm import LLM
from cacheclm.mab import Sample, load_sample
from cacheclm.report import write_report
from cacheclm.run import run_sample


def make_llm(cfg, price, budget):
    agent = cfg["agent"]
    key = os.environ[agent["api_key_env"]] if "api_key_env" in agent else "local"  # a local server needs no key
    client = openai.OpenAI(base_url=agent["base_url"], api_key=key, max_retries=0, timeout=agent.get("timeout", 300))
    return LLM(agent["model"], client, cfg["cache_dir"], cfg["temperature"], price, agent.get("options"), budget.spend)


def jobs(cfg, smoke):
    """[(sample spec, repeat, arms)]: primary arms first, then reference arms once per full-run sample, so a budget stop
    costs references before it costs primary runs. The smoke runs every arm, so the references are checked live once,
    unless a smoke sample names its own `arms`."""
    if smoke:
        return [(s, 0, tuple(s.get("arms", ARMS + REFERENCES))) for s in cfg["smoke"]["samples"]]
    primary = [(s, r, ARMS) for r in cfg["repeats"] for s in cfg["samples"]]
    extra = [(s, s["repeat"], ARMS) for s in cfg.get("extra_runs", [])]
    return primary + extra + [(s, 0, REFERENCES) for s in cfg["samples"]]


def load(cfg, spec):
    """The sample a spec names, cut to [start:end] when given (the EventQA smoke uses text no evaluated row covers),
    with its questions starting at question_start, or picked by question_ids, when given (the ones about that text)."""
    sample = load_sample(cfg["data_dir"], spec["split"], spec["row"])
    if "start" in spec:
        sample = Sample(f"{sample.sid}[{spec['start']}:{spec['end']}]", sample.source,
                        sample.context[spec["start"]:spec["end"]], sample.questions, sample.answers)
    if "question_start" in spec:
        q = spec["question_start"]
        sample = Sample(f"{sample.sid}q{q}", sample.source, sample.context, sample.questions[q:], sample.answers[q:])
    if "question_ids" in spec:  # questions located in the text shown, when they are not one contiguous run
        ids = spec["question_ids"]
        sample = Sample(f"{sample.sid}q{ids[0]}-{ids[-1]}x{len(ids)}", sample.source, sample.context,
                        [sample.questions[i] for i in ids], [sample.answers[i] for i in ids])
    return sample


def run_config(cfg, spec):
    """The config with the spec's own settings (any key the config has, such as context_budget) put over it."""
    return {**cfg, **{k: v for k, v in spec.items() if k in cfg}}


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
    try:
        for spec, repeat, arms in jobs(cfg, smoke):
            sample = load(cfg, spec)
            run_cfg = run_config(cfg, spec)
            for arm in arms:
                try:
                    acc = run_sample(sample, arm, repeat, llm, run_cfg, prices["deepseek"], runs_dir, spec.get("questions"))
                except RuntimeError as e:  # API kept failing after retries: skip, the report lists it as incomplete
                    print(f"FAILED   {arm:8s} {sample.sid:28s} r{repeat}: {e}")
                    continue
                print(f"{arm:8s} {sample.sid:28s} r{repeat}  accuracy {acc:.2f}  total spend ${budget.total:.3f}")
    except BudgetExceeded as e:
        print(f"Stopped: {e}. Finished runs are kept; raise budget_usd in the config to continue.")


if __name__ == "__main__":
    main()
