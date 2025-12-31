import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIG ---
MANIFEST_PATH = "study_manifest_v1.csv"

def run_eda():
    if not os.path.exists(MANIFEST_PATH):
        print("Error: Run manifest.py first!")
        return

    df = pd.read_csv(MANIFEST_PATH)
    
    # 1. Basic Stats
    total_subs = len(df)
    # DX_GROUP: 1=ASD, 2=Control
    asd_count = len(df[df['DX_GROUP'] == 1])
    tc_count = len(df[df['DX_GROUP'] == 2])
    
    print("-" * 30)
    print("DATA QUALITY & STATISTICS REPORT")
    print("-" * 30)
    print(f"Total Subjects: {total_subs}")
    print(f"Total Images: {total_subs * 3}")
    print(f"Autism (ASD): {asd_count} ({asd_count/total_subs:.1%})")
    print(f"Typical Control (TC): {tc_count} ({tc_count/total_subs:.1%})")
    print("-" * 30)

    # 2. Visualizations
    plt.figure(figsize=(15, 5))

    # Plot 1: Diagnosis Distribution (Phase 2.3 Task)
    plt.subplot(1, 3, 1)
    sns.countplot(x='DX_GROUP', data=df, palette='viridis')
    plt.title('Class Distribution (1:ASD, 2:TC)')
    plt.xticks([0, 1], ['ASD', 'Control'])

    # Plot 2: Age Distribution (Phase 2.3 Task)
    plt.subplot(1, 3, 2)
    sns.histplot(df['AGE_AT_SCAN'], kde=True, color='blue')
    plt.title('Age Distribution')

    # Plot 3: Site Distribution (Phase 2.2 Task)
    plt.subplot(1, 3, 3)
    df['SITE_ID'].value_counts().plot(kind='bar')
    plt.title('Subjects per Site')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('eda_report.png')
    print("EDA Visualizations saved to: eda_report.png")

    # 3. Create Split Statistics (Phase 2.2 Deliverable)
    split_stats = df.groupby(['Split', 'DX_GROUP']).size().unstack()
    print("\nSplit Distribution (Stratification Check):")
    print(split_stats)

if __name__ == "__main__":
    run_eda()