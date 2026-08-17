# wx:motivation-random mask
#!/usr/bin/env python3
"""Collect task success rates from experiment logs and export them to Excel.

Expected directory fragments:
    master_seed-20260814/keep_ratio-0.99/mask_seed-883693/
    google_robot_pick_coke_can/000_success_0.82.log

The script extracts the final line whose metric name is exactly ``success``
(so ``final_success`` is not mistakenly selected). Every seed is retained as
an independent row; no averaging is applied.
"""

from __future__ import annotations

import argparse
import re
from copy import copy
from pathlib import Path

import pandas as pd


KEEP_RATIO_RE = re.compile(r"^keep[_-]ratio-(.+)$")
MASTER_SEED_RE = re.compile(r"^master[_-]seed-(.+)$")
MASK_SEED_RE = re.compile(r"^mask[_-]seed-(.+)$")

# Matches an exact metric named "success" after the log separator " - ".
# It intentionally does not match "final_success".
SUCCESS_RE = re.compile(
    r"(?:^|\s-\s)success\s*:\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$",
    re.MULTILINE,
)


def nearest_tag(path: Path, pattern: re.Pattern[str]) -> str | None:
    """Return the nearest matching ancestor tag value."""
    for parent in path.parents:
        match = pattern.match(parent.name)
        if match:
            return match.group(1)
    return None


def parse_success(log_path: Path) -> float | None:
    """Read the last exact ``success: value`` metric in a log file."""
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = SUCCESS_RE.findall(text)
    return float(matches[-1]) if matches else None


def ratio_sort_key(value: str) -> tuple[int, float | str]:
    """Sort numeric keep ratios numerically, other labels lexicographically."""
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def collect(results_root: Path) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, object]] = []
    warnings: list[str] = []

    for log_path in sorted(results_root.rglob("*.log")):
        keep_ratio = nearest_tag(log_path, KEEP_RATIO_RE)
        mask_seed = nearest_tag(log_path, MASK_SEED_RE)
        master_seed = nearest_tag(log_path, MASTER_SEED_RE)

        # Ignore unrelated logs outside the requested experiment hierarchy.
        if keep_ratio is None or mask_seed is None:
            continue

        success = parse_success(log_path)
        if success is None:
            warnings.append(f"No exact 'success:' metric: {log_path}")
            continue

        # Under the shown layout, the task directory is the log's parent.
        task = log_path.parent.name
        rows.append(
            {
                "task": task,
                "keep_ratio": keep_ratio,
                "success_rate": success,
                "master_seed": master_seed or "",
                "mask_seed": mask_seed,
                "log_file": str(log_path),
            }
        )

    columns = [
        "task",
        "keep_ratio",
        "success_rate",
        "master_seed",
        "mask_seed",
        "log_file",
    ]
    return pd.DataFrame(rows, columns=columns), warnings


def export_excel(details: pd.DataFrame, output_path: Path) -> None:
    if details.empty:
        raise RuntimeError(
            "No valid results found. Check --results-root and directory names."
        )

    ratio_order = sorted(details["keep_ratio"].unique(), key=ratio_sort_key)

    # Long format is directly usable by seaborn/matplotlib. Each row is one
    # independent seed result, never an average across seeds.
    details = details.copy()
    details["keep_ratio_numeric"] = pd.to_numeric(
        details["keep_ratio"], errors="coerce"
    )
    details["mask_ratio_numeric"] = 1.0 - details["keep_ratio_numeric"]
    details["run_id"] = (
        "master_seed-"
        + details["master_seed"].astype(str)
        + "/mask_seed-"
        + details["mask_seed"].astype(str)
    )
    details["keep_ratio"] = pd.Categorical(
        details["keep_ratio"], categories=ratio_order, ordered=True
    )
    details = details.sort_values(
        ["task", "keep_ratio", "master_seed", "mask_seed", "log_file"]
    )

    plot_data = details[
        [
            "task",
            "keep_ratio",
            "keep_ratio_numeric",
            "mask_ratio_numeric",
            "success_rate",
            "master_seed",
            "mask_seed",
            "run_id",
        ]
    ].copy()

    # A second seed-preserving view: each task/seed is a row and each keep ratio
    # is a column. pivot (not pivot_table) deliberately rejects duplicates
    # instead of silently averaging them.
    duplicate_key = ["task", "master_seed", "mask_seed", "keep_ratio"]
    if details.duplicated(duplicate_key).any():
        seed_matrix = pd.DataFrame(
            {
                "note": [
                    "Seed_Matrix was not generated because duplicate logs exist "
                    "for the same task/master_seed/mask_seed/keep_ratio. "
                    "Use Plot_Data or Details; no values were averaged."
                ]
            }
        )
    else:
        seed_matrix = (
            details.pivot(
                index=["task", "master_seed", "mask_seed"],
                columns="keep_ratio",
                values="success_rate",
            )
            .reindex(columns=ratio_order)
            .reset_index()
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        plot_data.to_excel(writer, sheet_name="Plot_Data", index=False)
        seed_matrix.to_excel(writer, sheet_name="Seed_Matrix", index=False)
        details.to_excel(writer, sheet_name="Details", index=False)

        for sheet_name, worksheet in writer.sheets.items():
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                bold_font = copy(cell.font)
                bold_font.bold = True
                cell.font = bold_font
            for column_cells in worksheet.columns:
                width = min(
                    max(len(str(cell.value or "")) for cell in column_cells) + 2,
                    60,
                )
                worksheet.column_dimensions[column_cells[0].column_letter].width = width

        # Store success rates as percentages while keeping values numeric.
        for worksheet in writer.sheets.values():
            for row in worksheet.iter_rows():
                for cell in row:
                    header = worksheet.cell(row=1, column=cell.column).value
                    if header in {
                        "success_rate",
                        "keep_ratio_numeric",
                        "mask_ratio_numeric",
                    }:
                        if isinstance(cell.value, (int, float)):
                            cell.number_format = "0.00%"
        matrix_ws = writer.sheets["Seed_Matrix"]
        for row in matrix_ws.iter_rows(min_row=2, min_col=2):
            for cell in row:
                if isinstance(cell.value, (int, float)):
                    cell.number_format = "0.00%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect exact success metrics from logs into an Excel workbook."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default='/data/Xixixi/VLA/PCD/results/motivation-50/random_mask/open_pi_zero/target-multi_modal_projector',
        help="Root directory to search recursively, e.g. results/motivation-50/random_mask",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/Xixixi/VLA/PCD/results/motivation-50/random_mask/open_pi_zero/target-multi_modal_projector/success_rates.xlsx"),
        help="Output Excel path (default: success_rates.xlsx)",
    )
    args = parser.parse_args()

    root = args.results_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Results root does not exist or is not a directory: {root}")

    details, warnings = collect(root)
    export_excel(details, args.output.expanduser().resolve())

    print(f"Collected {len(details)} independent seed result(s); no averaging applied.")
    print(f"Excel saved to: {args.output.expanduser().resolve()}")
    if warnings:
        print(f"Skipped {len(warnings)} log(s) without exact success metrics:")
        for warning in warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
