import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

CSV_PATH   = "../results/simulation_benchmark_summary.csv"
OUTPUT_DIR = "../results/graphs_simulation/"

MODELS = ["BIOMISTRAL", "LLAMA3", "MISTRAL", "GEMMA2", "PHI3"]


# ==========================================
# HELPERS
# ==========================================
def label_bars_inside(ax, values, fmt="{:.1f}", threshold=0.6):
    """
    Place value labels inside bars (white, near top) so they never
    overflow the axis boundary or collide with the title.
    Falls back to black text just below top for very short bars.
    """
    ylim_max = ax.get_ylim()[1]
    for i, v in enumerate(values):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        inside = v > threshold * ylim_max
        y      = v - 0.25 if inside else v + 0.15
        va     = "top"    if inside else "bottom"
        color  = "white"  if inside else "black"
        ax.text(i, y, fmt.format(v), ha="center", va=va,
                fontweight="bold", fontsize=11, color=color)


def load_data():
    df = pd.read_csv(CSV_PATH, delimiter=';')
    numeric_cols = [
        "Heuristic Patient (/10)", "Heuristic Doctor (/10)", "Heuristic Total (/20)",
        "LLM Judge Total (/20)",
        "LLM Patient - Natural Language (/4)", "LLM Patient - Symptom Coherence (/3)",
        "LLM Patient - Role Adherence (/3)",   "LLM Patient - Total (/10)",
        "LLM Doctor - Question Quality (/4)",  "LLM Doctor - Diagnostic Accuracy (/3)",
        "LLM Doctor - Role Adherence (/3)",    "LLM Doctor - Total (/10)",
        "Patient Avg Words", "Patient Jargon/Turn", "Doctor Avg Words", "Doctor Questions",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"✅ Loaded {len(df)} rows from CSV.")
    return df


# ==========================================
# GRAPH 1 — Average heuristic scores by role
# Labels always inside bars → no title overlap.
# ==========================================
def plot_average_grades_by_role(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Average Heuristic Score per Model by Role",
                 fontsize=16, fontweight='bold')

    def model_avg(df, model, score_col, model_col):
        s = df[df[model_col] == model][score_col].dropna()
        return s.mean() if len(s) > 0 else 0

    patient_avgs = [model_avg(df, m, "Heuristic Patient (/10)", "Patient Model") for m in MODELS]
    ax1 = axes[0]
    sns.barplot(x=MODELS, y=patient_avgs, palette="viridis", ax=ax1)
    ax1.set_title("As PATIENT — Heuristic Score (/10)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Average Score / 10")
    ax1.set_ylim(0, 10)
    ax1.tick_params(axis='x', rotation=15)
    label_bars_inside(ax1, patient_avgs)

    doctor_avgs = [model_avg(df, m, "Heuristic Doctor (/10)", "Doctor Model") for m in MODELS]
    ax2 = axes[1]
    sns.barplot(x=MODELS, y=doctor_avgs, palette="magma", ax=ax2)
    ax2.set_title("As DOCTOR — Heuristic Score (/10)", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Average Score / 10")
    ax2.set_ylim(0, 10)
    ax2.tick_params(axis='x', rotation=15)
    label_bars_inside(ax2, doctor_avgs)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_average_grades_by_role.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 Graph 1 saved: 1_average_grades_by_role.png")


# ==========================================
# GRAPH 2 — 5×5 Heatmap, Heuristic Total
# ==========================================
def plot_heatmap_heuristic(df):
    matrix = pd.DataFrame(index=MODELS, columns=MODELS, dtype=float)
    for _, row in df.iterrows():
        p = str(row.get("Patient Model", "")).upper()
        d = str(row.get("Doctor Model",  "")).upper()
        v = row.get("Heuristic Total (/20)")
        if p in MODELS and d in MODELS and pd.notna(v):
            matrix.loc[p, d] = float(v)

    plt.figure(figsize=(9, 7))
    sns.heatmap(matrix.astype(float), annot=True, fmt=".1f",
                cmap="RdYlGn", vmin=0, vmax=20, linewidths=0.5,
                cbar_kws={'label': 'Heuristic Total (/20)'},
                annot_kws={"size": 12, "weight": "bold"})
    plt.title("Patient × Doctor Heatmap — Heuristic Score (/20)",
              fontsize=15, fontweight='bold', pad=20)
    plt.ylabel("Patient Model", fontsize=13)
    plt.xlabel("Doctor Model",  fontsize=13)
    plt.xticks(rotation=15)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_heatmap_heuristic.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 Graph 2 saved: 2_heatmap_heuristic.png")


# ==========================================
# GRAPH 3 — 5×5 Heatmap, LLM Judge Total
# ==========================================
def plot_heatmap_llm_judge(df):
    if df["LLM Judge Total (/20)"].isna().all():
        print("⚠️  Graph 3 skipped: no LLM judge data (run without --no-judge).")
        return

    matrix = pd.DataFrame(index=MODELS, columns=MODELS, dtype=float)
    for _, row in df.iterrows():
        p = str(row.get("Patient Model", "")).upper()
        d = str(row.get("Doctor Model",  "")).upper()
        v = row.get("LLM Judge Total (/20)")
        if p in MODELS and d in MODELS and pd.notna(v):
            matrix.loc[p, d] = float(v)

    plt.figure(figsize=(9, 7))
    sns.heatmap(matrix.astype(float), annot=True, fmt=".1f",
                cmap="RdYlGn", vmin=0, vmax=20, linewidths=0.5,
                cbar_kws={'label': 'LLM Judge Total (/20)'},
                annot_kws={"size": 12, "weight": "bold"})
    plt.title("Patient × Doctor Heatmap — LLM Judge Score (/20)",
              fontsize=15, fontweight='bold', pad=20)
    plt.ylabel("Patient Model", fontsize=13)
    plt.xlabel("Doctor Model",  fontsize=13)
    plt.xticks(rotation=15)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_heatmap_llm_judge.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 Graph 3 saved: 3_heatmap_llm_judge.png")


# ==========================================
# GRAPH 4 — Boxplots by role (heuristic)
# ==========================================
def plot_boxplots_by_role(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Heuristic Score Distribution by Model and Role",
                 fontsize=16, fontweight='bold')

    def build_role_df(df, score_col, model_col):
        rows = []
        for m in MODELS:
            for v in df[df[model_col] == m][score_col].dropna():
                rows.append({"Model": m, "Score": v})
        return pd.DataFrame(rows)

    for ax, score_col, model_col, title, palette in [
        (axes[0], "Heuristic Patient (/10)", "Patient Model",
         "PATIENT Role — Score Distribution (/10)", "Set2"),
        (axes[1], "Heuristic Doctor (/10)",  "Doctor Model",
         "DOCTOR Role — Score Distribution (/10)",  "Set3"),
    ]:
        role_df = build_role_df(df, score_col, model_col)
        if not role_df.empty:
            sns.boxplot( x="Model", y="Score", data=role_df, palette=palette, ax=ax)
            sns.swarmplot(x="Model", y="Score", data=role_df, color=".25", size=6, ax=ax)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_ylabel("Score / 10")
            ax.set_ylim(0, 11)
            ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_boxplots_by_role.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 Graph 4 saved: 4_boxplots_by_role.png")


# ==========================================
# GRAPH 5 — LLM Judge sub-scores: Patient + Doctor side by side
# Single graph replacing the two redundant separate ones.
# Each group of bars = one model; colours = criteria.
# Normalised to % of max so patient (/4,/3,/3) and doctor (/4,/3,/3)
# share the same y-axis and are directly comparable.
# ==========================================
def plot_llm_judge_subscores_combined(df):
    all_subcols = {
        # Patient criteria
        "Patient — Natural Language (/4)":   ("LLM Patient - Natural Language (/4)",   4),
        "Patient — Symptom Coherence (/3)":  ("LLM Patient - Symptom Coherence (/3)",  3),
        "Patient — Role Adherence (/3)":     ("LLM Patient - Role Adherence (/3)",      3),
        # Doctor criteria
        "Doctor — Question Quality (/4)":    ("LLM Doctor - Question Quality (/4)",     4),
        "Doctor — Diagnostic Accuracy (/3)": ("LLM Doctor - Diagnostic Accuracy (/3)", 3),
        "Doctor — Role Adherence (/3)":      ("LLM Doctor - Role Adherence (/3)",       3),
    }

    # Check at least some data is available
    has_data = any(
        col in df.columns and not df[col].isna().all()
        for col, _ in all_subcols.values()
    )
    if not has_data:
        print("⚠️  Graph 5 skipped: no LLM judge sub-score data found.")
        return

    records = []
    for label, (col, mx) in all_subcols.items():
        if col not in df.columns:
            continue
        # Use Patient Model column for patient criteria, Doctor Model for doctor
        model_col = "Patient Model" if label.startswith("Patient") else "Doctor Model"
        for m in MODELS:
            val = df[df[model_col] == m][col].dropna().mean()
            if not np.isnan(val):
                records.append({
                    "Model":     m,
                    "Criterion": label,
                    "Score %":   round((val / mx) * 100, 1),
                })

    if not records:
        print("⚠️  Graph 5 skipped: no computable sub-scores.")
        return

    plot_df = pd.DataFrame(records)

    # Split patient and doctor for two subplots — same scale (0-100%)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.suptitle("LLM Judge Sub-scores per Model — Patient & Doctor\n(normalised, % of criterion max)",
                 fontsize=15, fontweight='bold')

    for ax, prefix, title, palette in [
        (axes[0], "Patient", "PATIENT sub-scores", "Set2"),
        (axes[1], "Doctor",  "DOCTOR sub-scores",  "Set3"),
    ]:
        sub = plot_df[plot_df["Criterion"].str.startswith(prefix)].copy()
        # Shorten label for legend readability
        sub["Criterion"] = sub["Criterion"].str.replace(f"{prefix} — ", "", regex=False)
        if not sub.empty:
            sns.barplot(x="Model", y="Score %", hue="Criterion",
                        data=sub, palette=palette, ax=ax)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_ylabel("Score (% of max)" if ax is axes[0] else "")
            ax.set_ylim(0, 110)
            ax.tick_params(axis='x', rotation=15)
            ax.legend(title="Criterion", fontsize=9)
            # Reference line at 100%
            ax.axhline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "5_llm_judge_subscores_combined.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 Graph 5 saved: 5_llm_judge_subscores_combined.png")


# ==========================================
# GRAPH 6 — Heuristic vs LLM Judge per model and role
# Shows for each model the heuristic score and the LLM judge score
# on the same axis, split by role (patient / doctor).
# Makes it immediately visible whether the two systems agree
# and which model benefits most from semantic evaluation.
# ==========================================
def plot_heuristic_vs_llm_by_model(df):
    has_llm = "LLM Patient - Total (/10)" in df.columns and \
              not df["LLM Patient - Total (/10)"].isna().all()
    if not has_llm:
        print("⚠️  Graph 6 skipped: no LLM judge total scores found.")
        return

    records = []
    for m in MODELS:
        for role, heuristic_col, llm_col, model_col in [
            ("Patient", "Heuristic Patient (/10)", "LLM Patient - Total (/10)", "Patient Model"),
            ("Doctor",  "Heuristic Doctor (/10)",  "LLM Doctor - Total (/10)",  "Doctor Model"),
        ]:
            subset = df[df[model_col] == m]
            h_val  = subset[heuristic_col].dropna().mean() if heuristic_col in df.columns else np.nan
            l_val  = subset[llm_col].dropna().mean()       if llm_col       in df.columns else np.nan
            if not np.isnan(h_val):
                records.append({"Model": m, "Role": role,
                                 "Scoring": "Heuristic", "Score": h_val})
            if not np.isnan(l_val):
                records.append({"Model": m, "Role": role,
                                 "Scoring": "LLM Judge", "Score": l_val})

    if not records:
        print("⚠️  Graph 6 skipped: no data to plot.")
        return

    plot_df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.suptitle("Heuristic vs LLM Judge Score per Model (/10)\nby Role",
                 fontsize=15, fontweight='bold')

    for ax, role, palette in [
        (axes[0], "Patient", ["#4C72B0", "#DD8452"]),
        (axes[1], "Doctor",  ["#4C72B0", "#DD8452"]),
    ]:
        sub = plot_df[plot_df["Role"] == role]
        if not sub.empty:
            sns.barplot(x="Model", y="Score", hue="Scoring",
                        data=sub, palette=palette, ax=ax)
            ax.set_title(f"{role.upper()} — Heuristic vs LLM Judge (/10)",
                         fontsize=13, fontweight='bold')
            ax.set_ylabel("Average Score / 10" if ax is axes[0] else "")
            ax.set_ylim(0, 10)
            ax.tick_params(axis='x', rotation=15)
            ax.legend(title="Scoring method", fontsize=10)
            # Labels inside bars
            for container in ax.containers:
                ax.bar_label(container, fmt="%.1f", label_type="center",
                             fontsize=9, fontweight="bold", color="white",
                             padding=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "6_heuristic_vs_llm_by_model.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 Graph 6 saved: 6_heuristic_vs_llm_by_model.png")


# ==========================================
# GRAPH 7 — Top 10 combinations (heuristic)
# Labels always inside bars.
# ==========================================
def plot_top_combinations(df, top_n=10):
    df_s = (df[["Config", "Heuristic Total (/20)"]]
            .dropna()
            .sort_values("Heuristic Total (/20)", ascending=False)
            .head(top_n))

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="Config", y="Heuristic Total (/20)",
                     data=df_s, palette="viridis")
    plt.title(f"Top {top_n} Patient × Doctor Combinations — Heuristic Score (/20)",
              fontsize=15, fontweight='bold', pad=20)
    plt.ylabel("Heuristic Total / 20")
    plt.ylim(0, 20)
    plt.xticks(rotation=30, ha='right')
    label_bars_inside(ax, df_s["Heuristic Total (/20)"].tolist(),
                      threshold=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "7_top_combinations.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 Graph 7 saved: 7_top_combinations.png")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("📊 Generating simulation benchmark graphs...\n")

    df = load_data()

    plot_average_grades_by_role(df)           # 1 — avg heuristic by role
    plot_heatmap_heuristic(df)                # 2 — 5×5 heuristic heatmap
    plot_heatmap_llm_judge(df)                # 3 — 5×5 LLM judge heatmap
    plot_boxplots_by_role(df)                 # 4 — boxplots by role
    plot_llm_judge_subscores_combined(df)     # 5 — LLM sub-scores patient+doctor fused
    plot_heuristic_vs_llm_by_model(df)        # 6 — heuristic vs LLM per model and role
    plot_top_combinations(df, top_n=10)       # 7 — top combinations

    print(f"\n✅ Done! 7 graphs saved in: {OUTPUT_DIR}")