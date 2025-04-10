import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from collections import Counter
from typing import List, Dict, Set

def load_concept_files(directory: str, seeds: List[int]) -> Dict[int, Set[str]]:
    """
    Load concept names from CSV files for each seed.
    Returns a dictionary mapping seeds to sets of concept names.
    The CSV files have no headers, and we only care about the second column.
    """
    concepts_by_seed = {}
    for seed in seeds:
        file_path = Path(directory) / f"concept_names_seed{seed}.csv"
        try:
            df = pd.read_csv(file_path, header=None)
            concepts_by_seed[seed] = set(df[1].str.strip().values)  # Using index 1 for second column
        except FileNotFoundError:
            print(f"Warning: File not found for seed {seed}")
    return concepts_by_seed

def analyze_concepts(concepts_by_seed: Dict[int, Set[str]]) -> Dict[str, Set[str]]:
    """
    Analyze concepts across seeds to find:
    - Unique concepts (present in only one seed)
    - Common concepts (present in all seeds)
    - Frequent concepts (present in majority of seeds)
    """
    all_seeds = set(concepts_by_seed.keys())
    n_seeds = len(all_seeds)
    threshold_frequent = n_seeds // 2 + 1  # Majority threshold
    
    # Count occurrence of each concept
    concept_counts = Counter()
    for concepts in concepts_by_seed.values():
        concept_counts.update(concepts)
    
    # Categorize concepts
    unique_concepts = {concept for concept, count in concept_counts.items() if count == 1}
    common_concepts = {concept for concept, count in concept_counts.items() if count == n_seeds}
    frequent_concepts = {concept for concept, count in concept_counts.items() 
                        if count >= threshold_frequent}
    
    return {
        'unique': unique_concepts,
        'common': common_concepts,
        'frequent': frequent_concepts
    }

def plot_statistics(concepts_by_seed: Dict[int, Set[str]], analysis_results: Dict[str, Set[str]], output_dir: str):
    """
    Create visualizations for the concept analysis:
    1. Bar plot of concept counts per seed
    2. Bar plot of concept categories (unique, common, frequent)
    """
    # Use seaborn's default styling
    sns.set_theme()
    
    fig, ax = plt.subplots(figsize=(8, 6))

    # Category distribution plot
    categories = ['Unique', 'Common', 'Frequent']
    category_counts = [len(analysis_results['unique']), 
                      len(analysis_results['common']),
                      len(analysis_results['frequent'])]

    sns.barplot(x=categories, y=category_counts, ax=ax)
    ax.set_title('Distribution of Concept Categories')
    ax.set_ylabel('Number of Concepts')
    
    plt.tight_layout()
    plt.savefig(output_dir + '/concept_analysis.png')
    plt.close()

def save_results(analysis_results: Dict[str, Set[str]], output_dir: str):
    """
    Save analysis results to text files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for category, concepts in analysis_results.items():
        with open(output_path / f"{category}_concepts.txt", 'w') as f:
            for concept in sorted(concepts):
                f.write(f"{concept}\n")

def main():
    # Configuration
    base_dir = "/scratch/cbm/dncbm/SAE/SAEImg/cc3m/clip_RN50/out/lr0.0005_l1coeff3e-05_ef8_rf10_hookout_bs4096_epo200"
    seeds = [42, 1948, 360, 0, 10, 100, 2048] 
    output_dir = "/scratch/cbm/dncbm/concept_analysis_results"
    
    # Load and analyze data
    concepts_by_seed = load_concept_files(base_dir, seeds)
    if not concepts_by_seed:
        print("No data found. Please check the directory and seed values.")
        return
    
    analysis_results = analyze_concepts(concepts_by_seed)
    
    # Save results
    save_results(analysis_results, output_dir)
    
    # Create visualizations
    print('Creating visualizations...')
    plot_statistics(concepts_by_seed, analysis_results, output_dir)
    print(f"Results saved to: {output_dir}")
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"Total number of seeds analyzed: {len(concepts_by_seed)}")
    print(f"Unique concepts (present in only one seed): {len(analysis_results['unique'])}")
    print(f"Common concepts (present in all seeds): {len(analysis_results['common'])}")
    print(f"Frequent concepts (present in majority of seeds): {len(analysis_results['frequent'])}")

if __name__ == "__main__":
    main()