import argparse
import math
import re
import shlex
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend for consistent saving
import matplotlib.pyplot as plt

RESULT_FILE = Path(__file__).with_name("evaluation_results.md")
METRIC_NAMES = [
    "Top-1 ID accuracy",
    "Top-3 ID accuracy",
    "Top-10 ID accuracy",
    "Top-1 accuracy",
    "Top-3 accuracy",
    "Top-50 accuracy",
    "Top-100 accuracy",
]


def option_value(tokens: List[str], option: str) -> str:
    """Return the argument following `option` if it exists."""
    for i, tok in enumerate(tokens):
        if tok == option and i + 1 < len(tokens):
            return tokens[i + 1]
    return ""


def parse_command_models(command: str) -> List[str]:
    """Extract model names from the command (supports evaluate.py and evaluate_twostage.py)."""
    tokens = shlex.split(command)
    if len(tokens) < 3:
        return []
    if tokens[1].endswith("evaluate.py"):
        return [tokens[2]]
    if tokens[1].endswith("evaluate_twostage.py") and len(tokens) >= 4:
        return [tokens[2], tokens[3]]
    return []


def normalize_model_name(model: str, fine_tuned: bool) -> str:
    if model.startswith("miewid"):
        return "MiewID-FT" if fine_tuned else "MiewID-msv3"
    return model


def infer_set(dataset: str, current_set: str) -> str:
    ds_lower = dataset.lower()
    if "validation" in ds_lower:
        return "Validation"
    if current_set:
        return "Validation" if current_set.lower().startswith("validation") else "Full"
    return "Full"


def extract_features(model: str, feature_map: Dict[str, str], raw_model: str = "") -> str:
    """Pick feature count from known map or model naming convention."""
    if model in feature_map:
        return feature_map[model]
    if raw_model and raw_model in feature_map:
        return feature_map[raw_model]
    if model.startswith("aliked-"):
        suffix = model.split("-")[-1]
        if suffix.isdigit():
            return suffix
    if model.startswith("MegaDescriptor"):
        match = re.search(r"-(\d+)$", model)
        if match:
            return match.group(1)
    return ""


def display_model_name(name: str, raw: str, feat: str) -> str:
    lower = name.lower()
    if raw.startswith("aliked-"):
        suffix = raw.split("-")[-1]
        if suffix.isdigit():
            return f"ALIKED ({suffix})"
    if lower.startswith("aliked"):
        return f"ALIKED ({feat})" if feat else "ALIKED"
    if lower == "sift":
        return "SIFT"
    if lower.startswith("miewid"):
        return name
    return name


def display_similarity_name(sim: str) -> str:
    mapping = {
        "lightglue": "LightGlue",
        "classical": "Classical",
    }
    return mapping.get(sim.lower(), sim)


def parse_command_details(command: str, stage: int) -> Dict[str, object]:
    """Parse model, similarity, and dataset from a command string."""
    tokens = shlex.split(command)
    if len(tokens) < 3:
        return {}

    has_checkpoint = any(tok.startswith("--checkpoint") for tok in tokens)

    if tokens[1].endswith("evaluate.py"):
        model = tokens[2]
        similarity = ""
        for tok in tokens[3:]:
            if tok.startswith("-"):
                break
            similarity = tok
            break
        similarity = similarity or "lightglue"
        dataset = option_value(tokens, "--val_csv")
        norm_name = normalize_model_name(model, has_checkpoint)
        return {
            "model": norm_name,
            "raw_model": model,
            "similarity": similarity,
            "dataset": dataset,
        }

    if tokens[1].endswith("evaluate_twostage.py") and len(tokens) >= 4:
        model1, model2 = tokens[2], tokens[3]
        similarity1 = option_value(tokens, "--similarity1") or "cosine"
        similarity2 = option_value(tokens, "--similarity2") or "lightglue"
        dataset1 = option_value(tokens, "--stage1_csv")
        dataset2 = option_value(tokens, "--stage2_csv")
        norm1 = normalize_model_name(model1, has_checkpoint)
        norm2 = normalize_model_name(model2, False)
        details = {
            "stage_models": [norm1, norm2],
            "stage_models_raw": [model1, model2],
            "stage_similarities": [similarity1, similarity2],
            "datasets": [dataset1, dataset2],
        }
        if stage == 1:
            details.update(
                {
                    "model": norm1,
                    "raw_model": model1,
                    "similarity": similarity1,
                    "dataset": dataset1,
                    "stage": stage,
                }
            )
            return details
        if stage == 2:
            details.update(
                {
                    "model": norm2,
                    "raw_model": model2,
                    "similarity": similarity2,
                    "dataset": dataset2,
                    "stage": stage,
                }
            )
            return details
    return {}


def parse_results(text: str) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    feature_map: Dict[str, str] = {}

    current_set = ""
    stage_context = 0
    last_command = ""
    last_models: List[str] = []
    command_buffer = ""
    current_command_entries: List[int] = []

    lines = text.splitlines()
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        # Handle multi-line commands
        if command_buffer:
            command_buffer += " " + line.rstrip("\\")
            if not line.endswith("\\"):
                last_command = command_buffer
                last_models = parse_command_models(last_command)
                current_command_entries = []
                command_buffer = ""
            continue

        if line.startswith("## "):
            current_set = line[3:].strip().title()
            continue

        if line.lower().startswith("python "):
            command_buffer = line.rstrip("\\")
            if not line.endswith("\\"):
                last_command = command_buffer
                last_models = parse_command_models(last_command)
                current_command_entries = []
                command_buffer = ""
            continue

        if line.startswith("Stage 1"):
            stage_context = 1
            continue
        if line.startswith("Stage 2"):
            stage_context = 2
            continue

        if "max_num_keypoints=" in line:
            match = re.search(r"max_num_keypoints=(\d+)", line)
            if match:
                for model in last_models:
                    if "aliked" in model:
                        feature_map.setdefault(model, match.group(1))
            continue

        if line.startswith("final_in_features"):
            match = re.search(r"final_in_features\s+(\d+)", line)
            if match:
                for model in last_models:
                    if "miewid" in model:
                        feature_map.setdefault(model, match.group(1))
            continue

        if line.startswith("Wall-clock"):
            numbers = re.findall(r"[0-9]+\.?[0-9]*", line)
            time_val = " / ".join(numbers) if numbers else line.split("Wall-clock", 1)[1].strip()
            for idx in current_command_entries:
                entries[idx]["time"] = time_val
            current_command_entries.clear()
            continue

        # Metrics lines contain seven floats
        metric_values = re.findall(r"\d+\.\d+", line)
        if len(metric_values) == len(METRIC_NAMES):
            details = parse_command_details(last_command, stage_context)
            if not details:
                continue
            dataset_set = infer_set(details.get("dataset", ""), current_set)
            entry = {
                "model": details["model"],
                "raw_model": details["raw_model"],
                "similarity": details["similarity"],
                "set": dataset_set,
                "time": "",
                "metrics": [float(v) for v in metric_values],
            }
            if "stage_models" in details:
                entry["stage_models"] = details["stage_models"]
                entry["stage_models_raw"] = details["stage_models_raw"]
                entry["stage_similarities"] = details["stage_similarities"]
                entry["stage"] = details.get("stage", stage_context)
            entries.append(entry)
            current_command_entries.append(len(entries) - 1)
            continue

    return entries, feature_map


def format_table(
    entries: List[Dict[str, str]],
    feature_map: Dict[str, str],
    fmt: str = "md",
    add_max_kp_label: bool = False,
) -> str:
    header = ["Model (# features)", "Similarity", "Set", "Time (sec)", *METRIC_NAMES]

    def pct(val: float) -> str:
        return f"{val * 100:.1f}%"

    rows: List[List[str]] = []
    for entry in entries:
        if "stage_models" in entry:
            stage_models = []
            for name, raw in zip(entry["stage_models"], entry["stage_models_raw"]):
                feat = extract_features(name, feature_map, raw)
                stage_models.append(display_model_name(name, raw, feat))
            stage_sims = [display_similarity_name(s) for s in entry["stage_similarities"] if s.lower() != "cosine"]
            tail_sim = stage_sims[-1] if stage_sims else ""
            parts = stage_models + ([tail_sim] if tail_sim else [])
            model_display = "+".join(parts)
            if entry.get("stage"):
                model_display = f"{model_display} (Stage{entry['stage']})"
            similarity_display = ", ".join(display_similarity_name(s) for s in entry["stage_similarities"])
        else:
            feat = extract_features(entry["model"], feature_map, entry.get("raw_model", ""))
            model_display = display_model_name(entry["model"], entry.get("raw_model", ""), feat)
            similarity_display = display_similarity_name(entry["similarity"])
            if add_max_kp_label and entry.get("raw_model", "").startswith("aliked-") and similarity_display.lower() == "lightglue":
                label_feat = extract_features(entry["model"], feature_map, entry.get("raw_model", ""))
                label_feat = label_feat or entry.get("raw_model", "").split("-")[-1]
                model_display = f"ALIKED({label_feat})+LightGlue"

        metric_vals = [pct(v) for v in entry["metrics"]]
        rows.append(
            [
                model_display,
                similarity_display,
                entry["set"],
                format_time(entry["time"]),
                *metric_vals,
            ]
        )

    if fmt == "tsv":
        return "\n".join(["\t".join(header)] + ["\t".join(row) for row in rows])

    # Default markdown
    sep = ["---"] * len(header)
    md_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for row in rows:
        md_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(md_lines)


def parse_time_seconds(time_str: str) -> float:
    """Parse a time string like '993.75 / 1036.31' and return the first numeric value."""
    matches = re.findall(r"[0-9]+\.?[0-9]*", time_str)
    if not matches:
        return float("nan")
    try:
        return float(matches[0])
    except ValueError:
        return float("nan")


def parse_max_time(time_str: str) -> float:
    """Return the maximum numeric time in the string."""
    matches = re.findall(r"[0-9]+\.?[0-9]*", time_str)
    if not matches:
        return float("nan")
    try:
        return max(float(m) for m in matches)
    except ValueError:
        return float("nan")


def format_time(time_str: str) -> str:
    """Format time with thousand separators using the max value in the string."""
    val = parse_max_time(time_str)
    if val != val:  # NaN check
        return ""
    return f"{val:,.2f}"


def plot_scatter(
    entries: List[Dict[str, str]],
    feature_map: Dict[str, str],
    out_file: Path,
    add_max_kp_label: bool = False,
) -> None:
    """Create a scatter plot of time vs Top-1 accuracy."""

    def build_label(entry: Dict[str, object]) -> Tuple[str, object]:
        if "stage_models" in entry:
            stage_models = []
            for name, raw in zip(entry["stage_models"], entry["stage_models_raw"]):
                display = display_model_name(name, raw, "")
                stage_models.append(display)
            stage_sims = [display_similarity_name(s) for s in entry["stage_similarities"] if s.lower() != "cosine"]
            tail_sim = stage_sims[-1] if stage_sims else ""
            parts = stage_models + ([tail_sim] if tail_sim else [])
            label = "+".join([re.sub(r" \([^)]*\)", "", p) for p in parts])
            return label, entry.get("stage")
        label_base = re.sub(
            r" \([^)]*\)",
            "",
            display_model_name(entry["model"], entry.get("raw_model", ""), ""),
        )
        similarity_display = display_similarity_name(entry["similarity"])
        if add_max_kp_label and entry.get("raw_model", "").startswith("aliked") and similarity_display.lower() == "lightglue":
            feat = extract_features(entry["model"], feature_map, entry.get("raw_model", ""))
            feat = feat or entry.get("raw_model", "").split("-")[-1]
            return feat, None  # only the max_num_keypoints value as label
        if similarity_display.lower() == "cosine":
            return label_base, None
        return f"{label_base}+{similarity_display}", None

    # Collect per-label data across sets (in minutes)
    data: Dict[str, Dict[str, Dict[str, float]]] = {}
    for entry in entries:
        set_key = entry.get("set", "").lower()
        if set_key not in {"full", "validation"}:
            continue
        if add_max_kp_label:
            # keep only ALIKED + LightGlue
            if not (
                entry.get("raw_model", "").startswith("aliked")
                and display_similarity_name(entry["similarity"]).lower() == "lightglue"
            ):
                continue
            if "stage_models" in entry:
                continue  # drop two-stage entries
        label, stage_id = build_label(entry)
        t_val_raw = parse_max_time(entry.get("time", ""))
        t_val = t_val_raw / 60.0 if t_val_raw == t_val_raw else None  # minutes, allow missing
        y_val = entry["metrics"][0] * 100.0  # Top-1 ID accuracy
        current = data.setdefault(label, {})
        existing = current.get(set_key)
        # Prefer entries with time, then higher stage, then higher accuracy
        replace = False
        if existing is None:
            replace = True
        else:
            if existing.get("time") is None and t_val is not None:
                replace = True
            elif (stage_id or 0) > existing.get("stage", 0):
                replace = True
            elif y_val > existing.get("acc", -1):
                replace = True
        if replace:
            current[set_key] = {"time": t_val, "acc": y_val, "stage": stage_id or 0}

    xs, ys, labels, colors = [], [], [], []
    for label, sets in data.items():
        # Special mixing: use Full time with Validation accuracy for MiewID-FT and two-stage
        if not add_max_kp_label and label in {"MiewID-FT", "MiewID-FT+ALIKED+LightGlue"}:
            full = sets.get("full", {})
            val = sets.get("validation", {})
            time_val = full.get("time") if full.get("time") is not None else val.get("time")
            # For two-stage marker, use full-set accuracy if available; else fall back to validation
            acc_val = full.get("acc") if full.get("acc") is not None else val.get("acc")
            if time_val is None or acc_val is None:
                continue
            x, y = time_val, acc_val
        else:
            full = sets.get("full", {})
            val = sets.get("validation", {})
            if "time" in full and full.get("time") is not None:
                x = full["time"]
                y = val.get("acc") if add_max_kp_label and "acc" in val else full.get("acc")
            elif "time" in val and val.get("time") is not None:
                x = val["time"]
                y = val.get("acc")
            else:
                continue
            if y is None:
                continue

        if y < 20.0:
            continue
        xs.append(x)
        ys.append(y)
        labels.append(label)
        colors.append("black")

    if not xs:
        print("No time data found to plot.")
        return

    plt.figure(figsize=(8, 8))
    plt.scatter(xs, ys, marker="o", s=100, color=colors)
    threshold_min = 2.0  # minutes threshold for label placement
    for x, y, label, c in zip(xs, ys, labels, colors):
        align_right = x <= threshold_min  # right-of-marker text when x <= 2 minutes
        offset_x = 10 if align_right else -10
        ha = "left" if align_right else "right"
        plt.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(offset_x, 2),
            fontsize=12,
            ha=ha,
            va="center",
            color=c,
            clip_on=False,
        )
    plt.xlabel("Running time, minutes", fontsize=13)
    plt.ylabel("Top-1 identity accuracy", fontsize=13)
    ax = plt.gca()
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.grid(True, axis="both", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d%%"))

    if add_max_kp_label:
        tick_vals = list(range(150, 451, 50))
        x_min, x_max = min(xs), max(xs)
        plt.xlim(min(0.9 * tick_vals[0], x_min * 0.9), max(tick_vals[-1] * 1.1, x_max * 1.05))
        plt.xticks(tick_vals)
        y_min, y_max = min(ys), max(ys)
        plt.ylim(max(70, y_min - 2), min(100, y_max + 2))
    else:
        plt.setp(ax, xticks=[0.5, 1, 10, 100, 500])
        plt.xlim(0.5, 500)
        plt.ylim(75, 100)
    plt.margins(x=0.05, y=0.05)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    print(f"Saved plot to {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize evaluation_results.md")
    parser.add_argument(
        "--format",
        choices=["md", "tsv"],
        default="md",
        help="Output format: markdown (md) or tab-separated (tsv) for easy Word import",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const=Path("results/evaluation_results.pdf"),
        type=Path,
        help="Optional path to save a scatter plot (default: results/evaluation_results.pdf)",
    )
    parser.add_argument(
        "--max_num_keypoints",
        action="store_true",
        help="Annotate ALIKED-xxxx runs as ALIKED(xxxx)+LightGlue in table and plot",
    )
    args = parser.parse_args()

    text = RESULT_FILE.read_text(encoding="utf-8")
    entries, feature_map = parse_results(text)
    table = format_table(entries, feature_map, fmt=args.format, add_max_kp_label=args.max_num_keypoints)
    print(table)

    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot_scatter(entries, feature_map, args.plot, add_max_kp_label=args.max_num_keypoints)


if __name__ == "__main__":
    main()
