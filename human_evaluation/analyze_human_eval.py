# -*- coding: utf-8 -*-
"""
Human Evaluation — Bước 3: Phân tích kết quả đánh giá
Author: Khiem
Run:  python human_evaluation/analyze_human_eval.py

Script này sẽ:
1. Load annotations từ tất cả annotator
2. Tính mean Adequacy, Fluency scores
3. Tính Cohen's Kappa (inter-annotator agreement)
4. Phân tích theo nhóm độ dài câu
5. Xuất báo cáo kết quả
"""

import sys, os
import json
import math
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# ============================================================
# CONFIG
# ============================================================
EVAL_DIR = os.path.join(BASE_DIR, "human_evaluation")
SAMPLES_FILE = os.path.join(EVAL_DIR, "eval_samples.json")
ANNOTATIONS_DIR = os.path.join(EVAL_DIR, "annotations")
RESULTS_FILE = os.path.join(EVAL_DIR, "human_eval_results.json")


# ============================================================
# Cohen's Kappa
# ============================================================
def cohens_kappa(ratings1, ratings2, n_categories=5):
    """
    Tính Cohen's Kappa giữa 2 annotator.

    Args:
        ratings1: list of int (scores từ annotator 1)
        ratings2: list of int (scores từ annotator 2)
        n_categories: số category (1-5)
    Returns:
        kappa: float
    """
    assert len(ratings1) == len(ratings2), "Hai annotator phải chấm cùng số câu"
    n = len(ratings1)
    if n == 0:
        return 0.0

    # Tạo confusion matrix
    categories = list(range(1, n_categories + 1))
    matrix = defaultdict(int)
    for r1, r2 in zip(ratings1, ratings2):
        matrix[(r1, r2)] += 1

    # p_o: observed agreement
    p_o = sum(matrix[(c, c)] for c in categories) / n

    # p_e: expected agreement by chance
    p_e = 0
    for c in categories:
        p1 = sum(matrix[(c, c2)] for c2 in categories) / n
        p2 = sum(matrix[(c1, c)] for c1 in categories) / n
        p_e += p1 * p2

    # Kappa
    if p_e == 1.0:
        return 1.0
    kappa = (p_o - p_e) / (1 - p_e)
    return kappa


def kappa_interpretation(kappa):
    """Diễn giải Cohen's Kappa."""
    if kappa < 0.20:
        return "Poor"
    elif kappa < 0.40:
        return "Fair"
    elif kappa < 0.60:
        return "Moderate"
    elif kappa < 0.80:
        return "Substantial"
    else:
        return "Almost Perfect"


# ============================================================
# Analysis
# ============================================================
def load_all_annotations(direction="envi"):
    """Load tất cả annotations cho một direction."""
    annotations = {}

    if not os.path.exists(ANNOTATIONS_DIR):
        print(f"  ERROR: Thư mục annotations không tồn tại: {ANNOTATIONS_DIR}")
        return annotations

    for fname in os.listdir(ANNOTATIONS_DIR):
        if not fname.endswith(".json"):
            continue
        if f"_{direction}.json" not in fname:
            continue

        path = os.path.join(ANNOTATIONS_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        annotator = data.get("annotator", fname.replace(f"_{direction}.json", ""))
        annotations[annotator] = {int(k): v for k, v in data.get("annotations", {}).items()}
        print(f"  Loaded {len(annotations[annotator])} annotations from {annotator}")

    return annotations


def analyze_direction(direction="envi"):
    """Phân tích kết quả cho một direction."""
    print(f"\n{'=' * 60}")
    print(f"  ANALYZING: {direction.upper()}")
    print(f"{'=' * 60}")

    # Load samples
    with open(SAMPLES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = {s["id"]: s for s in data["samples"]}

    # Load annotations
    all_annotations = load_all_annotations(direction)

    if not all_annotations:
        print(f"  Không tìm thấy annotations cho {direction.upper()}")
        return None

    annotator_names = list(all_annotations.keys())
    print(f"\n  Annotators: {annotator_names}")

    # ── Per-annotator stats ──
    results = {
        "direction": direction,
        "annotators": annotator_names,
        "per_annotator": {},
        "overall": {},
    }

    for name in annotator_names:
        annots = all_annotations[name]
        adequacy_scores = [v["adequacy"] for v in annots.values()]
        fluency_scores = [v["fluency"] for v in annots.values()]

        if not adequacy_scores:
            continue

        stats = {
            "count": len(annots),
            "adequacy_mean": sum(adequacy_scores) / len(adequacy_scores),
            "adequacy_std": _std(adequacy_scores),
            "fluency_mean": sum(fluency_scores) / len(fluency_scores),
            "fluency_std": _std(fluency_scores),
            "overall_mean": (sum(adequacy_scores) + sum(fluency_scores)) / (2 * len(adequacy_scores)),
        }

        # Phân tích theo length group
        length_stats = defaultdict(lambda: {"adequacy": [], "fluency": []})
        for sid, ann in annots.items():
            if sid in samples:
                group = samples[sid]["length_group"]
                length_stats[group]["adequacy"].append(ann["adequacy"])
                length_stats[group]["fluency"].append(ann["fluency"])

        stats["by_length"] = {}
        for group in ["short", "medium", "long"]:
            if group in length_stats:
                ls = length_stats[group]
                stats["by_length"][group] = {
                    "count": len(ls["adequacy"]),
                    "adequacy_mean": sum(ls["adequacy"]) / len(ls["adequacy"]),
                    "fluency_mean": sum(ls["fluency"]) / len(ls["fluency"]),
                }

        # Score distribution
        stats["adequacy_distribution"] = {str(i): adequacy_scores.count(i) for i in range(1, 6)}
        stats["fluency_distribution"] = {str(i): fluency_scores.count(i) for i in range(1, 6)}

        results["per_annotator"][name] = stats

        print(f"\n  [{name}] ({stats['count']} samples)")
        print(f"    Adequacy: {stats['adequacy_mean']:.2f} ± {stats['adequacy_std']:.2f}")
        print(f"    Fluency:  {stats['fluency_mean']:.2f} ± {stats['fluency_std']:.2f}")
        print(f"    Overall:  {stats['overall_mean']:.2f}")

        if stats["by_length"]:
            print(f"    By length:")
            for group in ["short", "medium", "long"]:
                if group in stats["by_length"]:
                    bl = stats["by_length"][group]
                    print(f"      {group:8s}: Adeq={bl['adequacy_mean']:.2f}, Flu={bl['fluency_mean']:.2f} (n={bl['count']})")

    # ── Overall stats (average across annotators) ──
    all_adequacy = []
    all_fluency = []
    for name in annotator_names:
        annots = all_annotations[name]
        all_adequacy.extend([v["adequacy"] for v in annots.values()])
        all_fluency.extend([v["fluency"] for v in annots.values()])

    if all_adequacy:
        results["overall"] = {
            "total_annotations": len(all_adequacy),
            "adequacy_mean": sum(all_adequacy) / len(all_adequacy),
            "adequacy_std": _std(all_adequacy),
            "fluency_mean": sum(all_fluency) / len(all_fluency),
            "fluency_std": _std(all_fluency),
            "overall_mean": (sum(all_adequacy) + sum(all_fluency)) / (len(all_adequacy) + len(all_fluency)),
        }

    # ── Inter-annotator agreement (Cohen's Kappa) ──
    if len(annotator_names) >= 2:
        print(f"\n  --- Inter-Annotator Agreement ---")
        a1_name, a2_name = annotator_names[0], annotator_names[1]
        a1 = all_annotations[a1_name]
        a2 = all_annotations[a2_name]

        # Tìm câu cả 2 đều chấm
        common_ids = sorted(set(a1.keys()) & set(a2.keys()))
        print(f"  Common samples: {len(common_ids)}")

        if len(common_ids) >= 5:
            adeq1 = [a1[sid]["adequacy"] for sid in common_ids]
            adeq2 = [a2[sid]["adequacy"] for sid in common_ids]
            flu1 = [a1[sid]["fluency"] for sid in common_ids]
            flu2 = [a2[sid]["fluency"] for sid in common_ids]

            kappa_adeq = cohens_kappa(adeq1, adeq2)
            kappa_flu = cohens_kappa(flu1, flu2)
            kappa_avg = (kappa_adeq + kappa_flu) / 2

            results["inter_annotator_agreement"] = {
                "annotators": [a1_name, a2_name],
                "common_samples": len(common_ids),
                "kappa_adequacy": round(kappa_adeq, 4),
                "kappa_fluency": round(kappa_flu, 4),
                "kappa_average": round(kappa_avg, 4),
                "interpretation": kappa_interpretation(kappa_avg),
            }

            print(f"  Cohen's Kappa (Adequacy): {kappa_adeq:.4f} ({kappa_interpretation(kappa_adeq)})")
            print(f"  Cohen's Kappa (Fluency):  {kappa_flu:.4f} ({kappa_interpretation(kappa_flu)})")
            print(f"  Average Kappa:            {kappa_avg:.4f} ({kappa_interpretation(kappa_avg)})")
        else:
            print(f"  Cần ít nhất 5 common samples để tính Kappa")
    else:
        print(f"\n  [INFO] Chỉ có 1 annotator — không tính Cohen's Kappa")
        print(f"  Mời thêm 1 người chấm để tính inter-annotator agreement")

    return results


def _std(values):
    """Standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


# ============================================================
# Report generation
# ============================================================
def generate_markdown_table(results_envi, results_vien=None):
    """Tạo bảng kết quả dạng Markdown để paste vào README."""
    print(f"\n{'=' * 60}")
    print("  MARKDOWN TABLE (copy vào README.md)")
    print(f"{'=' * 60}")

    print("\n### Human Evaluation Results\n")

    if results_envi and results_envi.get("overall"):
        o = results_envi["overall"]
        print("#### EN → VI\n")
        print("| Metric | Score |")
        print("|--------|-------|")
        print(f"| **Adequacy** (1-5) | **{o['adequacy_mean']:.2f}** ± {o['adequacy_std']:.2f} |")
        print(f"| **Fluency** (1-5) | **{o['fluency_mean']:.2f}** ± {o['fluency_std']:.2f} |")
        print(f"| **Overall** (1-5) | **{o['overall_mean']:.2f}** |")

        n_annotators = len(results_envi.get("annotators", []))
        total = o.get("total_annotations", 0) // max(n_annotators, 1)
        print(f"\n> Evaluated on {total} random test samples by {n_annotators} annotator(s).")

        if "inter_annotator_agreement" in results_envi:
            iaa = results_envi["inter_annotator_agreement"]
            print(f"> Inter-annotator agreement (Cohen's κ): {iaa['kappa_average']:.2f} ({iaa['interpretation']})")

    if results_vien and results_vien.get("overall"):
        o = results_vien["overall"]
        print("\n#### VI → EN\n")
        print("| Metric | Score |")
        print("|--------|-------|")
        print(f"| **Adequacy** (1-5) | **{o['adequacy_mean']:.2f}** ± {o['adequacy_std']:.2f} |")
        print(f"| **Fluency** (1-5) | **{o['fluency_mean']:.2f}** ± {o['fluency_std']:.2f} |")
        print(f"| **Overall** (1-5) | **{o['overall_mean']:.2f}** |")

    # By length group
    if results_envi:
        has_length = any(
            "by_length" in stats
            for stats in results_envi.get("per_annotator", {}).values()
        )
        if has_length:
            print("\n#### By Sentence Length (EN → VI)\n")
            print("| Length | Adequacy | Fluency | N |")
            print("|--------|----------|---------|---|")
            # Aggregate from first annotator
            first_annotator = list(results_envi["per_annotator"].values())[0]
            for group in ["short", "medium", "long"]:
                if group in first_annotator.get("by_length", {}):
                    bl = first_annotator["by_length"][group]
                    label = {"short": "Short (5-15w)", "medium": "Medium (16-30w)", "long": "Long (31+w)"}[group]
                    print(f"| {label} | {bl['adequacy_mean']:.2f} | {bl['fluency_mean']:.2f} | {bl['count']} |")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  HUMAN EVALUATION — RESULTS ANALYSIS")
    print("=" * 60)

    if not os.path.exists(SAMPLES_FILE):
        print(f"\n  ERROR: {SAMPLES_FILE} not found!")
        print("  Chạy sample_for_eval.py trước.")
        return

    # Analyze both directions
    results_envi = analyze_direction("envi")
    results_vien = analyze_direction("vien")

    # Save combined results
    combined = {
        "envi": results_envi,
        "vien": results_vien,
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {RESULTS_FILE}")

    # Generate markdown
    generate_markdown_table(results_envi, results_vien)

    print(f"\n{'=' * 60}")
    print("  DONE!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
