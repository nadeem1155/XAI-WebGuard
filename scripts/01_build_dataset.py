"""
Script 01: Build and Preprocess Multi-Class Dataset
Combines CSIC 2010, SQLi, and XSS datasets into a unified, balanced, and cleaned corpus.
Saves:
- data/cybersecurity_dataset.csv
- data/train.csv
- data/test.csv
- results/class_distribution.png
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import clean_and_normalize, extract_csic_requests


def main():
    print("=" * 70)
    print("PHASE 1: DATASET COLLECTION, CLEANING, AND PREPROCESSING")
    print("=" * 70)

    data_dir = os.path.join("data")
    raw_dir = os.path.join("data", "raw")
    results_dir = os.path.join("results")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    sqli_path = os.path.join(raw_dir, "sqli_dataset.csv")
    xss_path = os.path.join(raw_dir, "xss_dataset.csv")
    csic_path = os.path.join(raw_dir, "csic_final.csv")

    assert os.path.exists(sqli_path), f"Missing {sqli_path}"
    assert os.path.exists(xss_path), f"Missing {xss_path}"
    assert os.path.exists(csic_path), f"Missing {csic_path}"

    # 1. Process SQL Injection Dataset
    print("\n[1/4] Loading and cleaning SQL Injection dataset...")
    df_sqli = pd.read_csv(sqli_path, encoding="utf-8", on_bad_lines="skip")
    sqli_raw_len = len(df_sqli)
    print(f"  Raw SQLi records: {sqli_raw_len}")

    # SQLi attacks (label == 1)
    sqli_attacks_raw = df_sqli[df_sqli["Label"] == 1]["Query"].dropna().astype(str)
    sqli_attacks_clean = sqli_attacks_raw.apply(clean_and_normalize)
    sqli_attacks = sqli_attacks_clean[sqli_attacks_clean.str.len() >= 3].drop_duplicates()
    print(f"  Cleaned unique SQLi attacks: {len(sqli_attacks):,}")

    # SQLi benign queries (label == 0)
    sqli_benign_raw = df_sqli[df_sqli["Label"] == 0]["Query"].dropna().astype(str)
    sqli_benign_clean = sqli_benign_raw.apply(clean_and_normalize)
    sqli_benign = sqli_benign_clean[sqli_benign_clean.str.len() >= 3].drop_duplicates()
    print(f"  Cleaned unique SQLi benign queries: {len(sqli_benign):,}")

    # 2. Process XSS Dataset
    print("\n[2/4] Loading and cleaning XSS dataset...")
    df_xss = pd.read_csv(xss_path, encoding="utf-8", on_bad_lines="skip")
    xss_raw_len = len(df_xss)
    print(f"  Raw XSS records: {xss_raw_len}")

    # XSS attacks (label == 1)
    xss_attacks_raw = df_xss[df_xss["Label"] == 1]["Sentence"].dropna().astype(str)
    xss_attacks_clean = xss_attacks_raw.apply(clean_and_normalize)
    xss_attacks = xss_attacks_clean[xss_attacks_clean.str.len() >= 3].drop_duplicates()
    print(f"  Cleaned unique XSS attacks: {len(xss_attacks):,}")

    # XSS benign sentences (label == 0)
    xss_benign_raw = df_xss[df_xss["Label"] == 0]["Sentence"].dropna().astype(str)
    xss_benign_clean = xss_benign_raw.apply(clean_and_normalize)
    xss_benign = xss_benign_clean[xss_benign_clean.str.len() >= 3].drop_duplicates()
    print(f"  Cleaned unique XSS benign samples: {len(xss_benign):,}")

    # 3. Process CSIC 2010 Dataset
    print("\n[3/4] Loading and cleaning CSIC 2010 HTTP requests...")
    df_csic = pd.read_csv(csic_path, encoding="utf-8", on_bad_lines="skip")
    print(f"  Raw CSIC 2010 records: {len(df_csic)}")
    csic_valid_requests = extract_csic_requests(df_csic, target_class="Valid")
    print(f"  Cleaned unique CSIC 2010 valid requests: {len(csic_valid_requests):,}")

    # 4. Assemble Balanced Multi-Class Dataset
    print("\n[4/4] Assembling harmonized multi-class dataset...")
    # Target approximately 8,000 samples per class to maintain optimal balance & training speed
    n_target = min(len(xss_attacks), 8000)
    print(f"  Target sample size per class: {n_target:,}")

    # Sample Class 1: SQLi
    df_class_sqli = pd.DataFrame({
        "payload": sqli_attacks.sample(n=n_target, random_state=42),
        "label": 1,
        "category": "SQL Injection"
    })

    # Sample Class 2: XSS
    df_class_xss = pd.DataFrame({
        "payload": xss_attacks.sample(n=n_target, random_state=42),
        "label": 2,
        "category": "Cross-Site Scripting"
    })

    # Combine Benign Sources (CSIC valid + SQLi benign + XSS benign) to create rich, diverse normal traffic
    benign_pool = pd.concat([
        pd.Series(csic_valid_requests).sample(n=min(len(csic_valid_requests), 3500), random_state=42),
        sqli_benign.sample(n=min(len(sqli_benign), 3000), random_state=42),
        xss_benign.sample(n=min(len(xss_benign), 3000), random_state=42)
    ]).drop_duplicates()

    df_class_benign = pd.DataFrame({
        "payload": benign_pool.sample(n=n_target, random_state=42),
        "label": 0,
        "category": "Benign Request"
    })

    # Combine all 3 classes
    full_dataset = pd.concat([df_class_benign, df_class_sqli, df_class_xss], ignore_index=True)
    full_dataset = full_dataset.drop_duplicates(subset=["payload"]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Compute length features for statistics
    full_dataset["char_length"] = full_dataset["payload"].str.len()

    print("\nDataset Composition Summary:")
    print("-" * 50)
    for cat, count in full_dataset["category"].value_counts().items():
        pct = (count / len(full_dataset)) * 100
        print(f"  * {cat:<24}: {count:>6,} ({pct:.1f}%)")
    print("-" * 50)
    print(f"  Total Cleaned Records   : {len(full_dataset):,}")
    print(f"  Mean Request Length     : {full_dataset['char_length'].mean():.1f} characters")
    print(f"  Max Request Length      : {full_dataset['char_length'].max():,} characters")
    print(f"  Min Request Length      : {full_dataset['char_length'].min():,} characters")

    # Save complete dataset
    full_dataset_path = os.path.join(data_dir, "cybersecurity_dataset.csv")
    full_dataset.to_csv(full_dataset_path, index=False)
    print(f"\n[+] Full dataset saved to: {full_dataset_path}")

    # Stratified Train-Test Split (80% train, 20% test)
    train_df, test_df = train_test_split(
        full_dataset,
        test_size=0.20,
        random_state=42,
        stratify=full_dataset["label"]
    )

    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[+] Training set saved to : {train_path} ({len(train_df):,} samples)")
    print(f"[+] Testing set saved to  : {test_path} ({len(test_df):,} samples)")

    # 5. Generate Publication-Quality Visualizations
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot of class distribution
    palette = ["#2ecc71", "#e74c3c", "#3498db"]
    sns.countplot(
        data=full_dataset,
        x="category",
        ax=axes[0],
        palette=palette,
        hue="category",
        legend=False
    )
    axes[0].set_title("Dataset Class Distribution (Multi-Class)", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xlabel("Attack Category", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Number of Samples", fontsize=11, fontweight="bold")
    for p in axes[0].patches:
        height = p.get_height()
        axes[0].annotate(f"{int(height):,}",
                         (p.get_x() + p.get_width() / 2., height),
                         ha="center", va="center", xytext=(0, 6),
                         textcoords="offset points", fontsize=10, fontweight="bold")

    # Boxplot of payload lengths per class
    sns.boxplot(
        data=full_dataset,
        x="category",
        y="char_length",
        ax=axes[1],
        palette=palette,
        hue="category",
        showfliers=False
    )
    axes[1].set_title("Payload Length Distribution by Category (excluding outliers)", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_xlabel("Attack Category", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Character Length", fontsize=11, fontweight="bold")

    plt.tight_layout()
    chart_path = os.path.join(results_dir, "class_distribution.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[+] Class distribution visualization saved to: {chart_path}")
    print("\nPhase 1 Completed Successfully!\n")


if __name__ == "__main__":
    main()
