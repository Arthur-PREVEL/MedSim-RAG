import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Scientific aesthetic configuration
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

CSV_PATH = "../results/ultimate_benchmark_summary.csv"
OUTPUT_DIR = "../results/graphs/"

# Models to analyze
MODELS = ["LLAMA3", "MISTRAL", "BIOMISTRAL", "GEMMA2", "PHI3"]

def load_data():
    # Make sure the file name matches your actual CSV file name
    df = pd.read_csv(CSV_PATH, delimiter=';')
    # Shorten pathology names for cleaner graph labels
    df['Pathology (Short)'] = df['Pathology (Ground Truth)'].apply(lambda x: x[:20] + '...' if len(x) > 20 else x)
    return df

def plot_average_total_grades(df):
    plt.figure(figsize=(10, 6))
    
    # Extract total grade columns and calculate the mean
    cols = [f"{m} - Total Grade (/20)" for m in MODELS]
    averages = df[cols].mean()
    
    # Create the barplot
    ax = sns.barplot(x=MODELS, y=averages.values, palette="viridis")
    plt.title("Average Total Grade per Model (out of 20)", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Average Grade / 20")
    plt.ylim(0, 20)
    
    # Add value labels on top of the bars
    for i, v in enumerate(averages):
        ax.text(i, v + 0.3, f"{v:.1f}", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_average_grades.png"), dpi=300)
    plt.close()

def plot_heatmap(df):
    plt.figure(figsize=(12, 8))
    
    # Prepare data for the heatmap
    cols = [f"{m} - Total Grade (/20)" for m in MODELS]
    heatmap_data = df.set_index('Pathology (Short)')[cols]
    heatmap_data.columns = MODELS  # Rename columns for cleanliness
    
    # Heatmap with Red (bad) -> Green (good) color palette
    sns.heatmap(heatmap_data, annot=True, cmap="RdYlGn", vmin=0, vmax=20, 
                linewidths=.5, cbar_kws={'label': 'Grade out of 20'})
    
    plt.title("Clinical Case Evaluations Heatmap", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Clinical Case")
    plt.xlabel("Evaluating Model")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_heatmap.png"), dpi=300)
    plt.close()

def plot_boxplots(df):
    plt.figure(figsize=(10, 6))
    
    cols = [f"{m} - Total Grade (/20)" for m in MODELS]
    data_melted = df[cols].melt(var_name="Model", value_name="Grade")
    data_melted["Model"] = data_melted["Model"].apply(lambda x: x.split(" -")[0])
    
    sns.boxplot(x="Model", y="Grade", data=data_melted, palette="Set2")
    sns.swarmplot(x="Model", y="Grade", data=data_melted, color=".25", size=6) # Adds individual points
    
    plt.title("Grade Dispersion and Severity (Boxplot)", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Grade Distribution (/20)")
    plt.ylim(0, 21)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_boxplots_severity.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("📊 Generating graphs...")
    df = load_data()
    plot_average_total_grades(df)
    plot_heatmap(df)
    plot_boxplots(df)
    print(f"✅ Done! 3 graphs have been generated in the folder: {OUTPUT_DIR}")