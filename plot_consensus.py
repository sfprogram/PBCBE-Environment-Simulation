import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path


run_styles = ['-', '-', '-', '-', '-']
gene_colors = ['#E74C3C', '#8B4513', '#FF8C00', '#2ECC71', '#3498DB']


def load_and_analyze_data():

    try:
        import glob

        all_run_data = {}
        params = None

        print("Loading data from all ensemble runs")
        for run_id in range(5):
            csv_files = glob.glob(f"training data/consensus/ensemble_run_{run_id}_seed_*.csv")
            if not csv_files:
                print(f"No csv file {run_id}")
                continue

            ensemble_csv_file = csv_files[0]
            print(f"Loading file: {ensemble_csv_file}")
            df = pd.read_csv(ensemble_csv_file)

            all_run_data[run_id] = df
            print(f"  Run {run_id}: {len(df)} records, generations {df['generation'].min()}-{df['generation'].max()}")

            if params is None:
                try:
                    with open("training data/consensus/ensemble_params.json", 'r') as f:
                        params = json.load(f)
                except FileNotFoundError:
                    params = {
                        'FROG_POP': 'Unknown',
                        'CONSENSUS_THRESHOLD': 'Unknown',
                        'EVOLUTION_STEP_SIZE': 'Unknown'
                    }

        return None, params, all_run_data

    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def create_gene_evolution_plot(all_run_data, params):

    if len(all_run_data) == 0:
        print("No data loaded from any runs")
        return

    frog_data_runs = {}
    for run_id, run_data in all_run_data.items():
        if 'species' not in run_data.columns:

            continue

        frog_data = run_data[run_data['species'] == 'frog'].copy()
        if len(frog_data) > 0:
            frog_data_runs[run_id] = frog_data

    if len(frog_data_runs) == 0:
        return

    gene_columns = ['wall_gene', 'food_gene', 'snake_gene', 'grass_gene', 'water_gene']
    gene_labels = ['Wall Avoidance', 'Food Seeking', 'Snake Avoidance', 'Grass Seeking', 'Water Seeking']

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']

    output_dir = Path("plots/consensus/create_gene_evolution_plot")
    output_dir.mkdir(parents=True, exist_ok=True)

    for run_id, frog_data in frog_data_runs.items():
        generations = sorted(frog_data['generation'].unique())

        avg_genes = {}
        for gene in gene_columns:

            avg_genes[gene] = []
            for gen in generations:
                gen_data = frog_data[frog_data['generation'] == gen]
                avg_value = gen_data[gene].mean()
                avg_genes[gene].append(avg_value)

        if not avg_genes:
            continue

        plt.figure(figsize=(12, 8))

        plot_count = 0
        for i, (gene, label) in enumerate(zip(gene_columns, gene_labels)):
            if gene in avg_genes:
                plt.plot(generations, avg_genes[gene],
                         marker='o', linewidth=2.5, markersize=6,
                         color=colors[i], label=label, alpha=0.8)

                plot_count += 1


        if plot_count == 0:
            plt.close()
            continue

        plt.xlabel('Generation', fontsize=12, fontweight='bold')
        plt.ylabel('Average Gene Value', fontsize=12, fontweight='bold')
        plt.title(f'Ensemble Run {run_id}: Gene Evolution Over Time', fontsize=14, fontweight='bold', pad=20)

        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left', framealpha=0.9, fontsize=10)

        all_values = [val for gene_vals in avg_genes.values() for val in gene_vals]
        if all_values:
            y_min, y_max = min(all_values), max(all_values)
            y_range = y_max - y_min
            if y_range > 0:
                plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        plt.tight_layout()

        output_path = output_dir / f"gene_evolution_run_{run_id}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {output_path}")


def compare_gene_across_runs(all_run_data, params):

    if isinstance(all_run_data, pd.DataFrame):

        all_run_data = {0: all_run_data}

    if len(all_run_data) == 0:
        print("No data loaded from any runs")
        return

    frog_data_runs = {}
    for run_id, run_data in all_run_data.items():

        frog_data = run_data[run_data['species'] == 'frog'].copy()
        if len(frog_data) > 0:
            frog_data_runs[run_id] = frog_data

    if len(frog_data_runs) == 0:
        return

    gene_columns = ['wall_gene', 'food_gene', 'snake_gene', 'grass_gene', 'water_gene']
    gene_labels = ['Wall Avoidance', 'Food Seeking', 'Snake Avoidance', 'Grass Seeking', 'Water Seeking']

    run_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    base_output_dir = Path("plots/consensus/compare_gene_across_runs")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    for gene_idx, (gene_col, gene_label, gene_color) in enumerate(zip(gene_columns, gene_labels, gene_colors)):

        plt.figure(figsize=(12, 8))

        valid_runs = 0

        for run_id, run_data in frog_data_runs.items():
            if gene_col not in run_data.columns:
                continue

            generations = sorted(run_data['generation'].unique())

            avg_values = []
            for gen in generations:
                gen_data = run_data[run_data['generation'] == gen]
                avg_value = gen_data[gene_col].mean()
                avg_values.append(avg_value)

            if not avg_values:
                continue

            plt.plot(generations, avg_values,
                     marker='o', linewidth=2.5, markersize=6,
                     color=run_colors[run_id % len(run_colors)],
                     linestyle=run_styles[run_id % len(run_styles)],
                     label=f'Run {run_id}', alpha=0.8)

            final_gen = generations[-1]
            final_value = avg_values[-1]
            plt.scatter(final_gen, final_value,
                        marker='X', s=100,
                        color=run_colors[run_id % len(run_colors)],
                        edgecolors='black', linewidth=1.5,
                        alpha=1.0, zorder=5)

            valid_runs += 1

        if valid_runs == 0:
            print(f"No valid data found for gene {gene_label}")
            plt.close()
            continue

        plt.xlabel('Generation', fontsize=12, fontweight='bold')
        plt.ylabel(f'{gene_label} Gene Value', fontsize=12, fontweight='bold')
        plt.title(f'{gene_label}: Evolution Comparison Across Ensemble Runs',fontsize=14, fontweight='bold', pad=20)

        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', framealpha=0.9, fontsize=10)

        all_values = []
        for run_data in frog_data_runs.values():
            if gene_col in run_data.columns:
                all_values.extend(run_data[gene_col].tolist())

        if all_values:
            y_min, y_max = min(all_values), max(all_values)
            y_range = y_max - y_min
            if y_range > 0:
                plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)


        plt.tight_layout()

        output_path = base_output_dir / f"{gene_col}_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {output_path}")


def compare_consensus_confidence_across_runs(all_run_data, params):

    if isinstance(all_run_data, pd.DataFrame):

        all_run_data = {0: all_run_data}

    if len(all_run_data) == 0:
        print("No data loaded from any runs")
        return

    consensus_data_runs = {}
    for run_id, run_data in all_run_data.items():

        consensus_data = run_data[run_data['species'] == 'consensus'].copy()
        if len(consensus_data) > 0:
            consensus_data_runs[run_id] = consensus_data


    if len(consensus_data_runs) == 0:
        return

    gene_columns = ['wall_gene', 'food_gene', 'snake_gene', 'grass_gene', 'water_gene']
    gene_labels = ['Wall Avoidance', 'Food Seeking', 'Snake Avoidance', 'Grass Seeking', 'Water Seeking']

    run_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    base_output_dir = Path("plots/consensus/consensus_confidence_across_runs")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    for gene_idx, (gene_col, gene_label, gene_color) in enumerate(zip(gene_columns, gene_labels, gene_colors)):

        plt.figure(figsize=(12, 8))

        valid_runs = 0

        for run_id, consensus_data in consensus_data_runs.items():

            gene_consensus_data = consensus_data[consensus_data['id'] == gene_col]

            valid_generations = []
            confidence_values = []

            for gen in sorted(gene_consensus_data['generation'].unique()):
                gen_data = gene_consensus_data[gene_consensus_data['generation'] == gen]
                if len(gen_data) > 0:
                    confidence = gen_data['fitness_or_confidence'].iloc[0]

                    if confidence != 0.0:
                        valid_generations.append(gen)
                        confidence_values.append(confidence)

            if not confidence_values:
                continue

            plt.plot(valid_generations, confidence_values, marker='o', linewidth=2.5, markersize=6,
                     color=run_colors[run_id % len(run_colors)], linestyle=run_styles[run_id % len(run_styles)],
                     label=f'Run {run_id}', alpha=0.8)

            if valid_generations and confidence_values:
                final_gen = valid_generations[-1]
                final_confidence = confidence_values[-1]
                plt.scatter(final_gen, final_confidence, marker='X', s=100, color=run_colors[run_id % len(run_colors)],
                            edgecolors='black', linewidth=1.5, alpha=1.0, zorder=5)

            valid_runs += 1

        plt.xlabel('Generation', fontsize=12, fontweight='bold')
        plt.ylabel(f'{gene_label} Consensus Confidence', fontsize=12, fontweight='bold')
        plt.title(f'{gene_label}: Consensus Confidence Evolution Across Ensemble Runs', fontsize=14, fontweight='bold', pad=20)

        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', framealpha=0.9, fontsize=10)

        plt.ylim(-0.05, 1.05)

        plt.tight_layout()

        output_path = base_output_dir / f"{gene_col}_consensus_confidence.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {output_path}")



def plot_individual_frog_gene_evolution(all_run_data, params):

    if isinstance(all_run_data, pd.DataFrame):

        all_run_data = {0: all_run_data}

    if len(all_run_data) == 0:
        print("No data loaded from any runs!")
        return

    gene_columns = ['wall_gene', 'food_gene', 'snake_gene', 'grass_gene', 'water_gene']
    gene_labels = ['Wall Avoidance', 'Food Seeking', 'Snake Avoidance', 'Grass Seeking', 'Water Seeking']

    base_output_dir = Path("plots/consensus/plot_individual_frog_gene_evolution")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    for run_id, run_data in all_run_data.items():

        frog_data = run_data[run_data['species'] == 'frog'].copy()

        run_output_dir = base_output_dir / f"run_{run_id}"
        run_output_dir.mkdir(parents=True, exist_ok=True)

        frog_ids = sorted(frog_data['id'].unique())
        generations = sorted(frog_data['generation'].unique())


        for gene_idx, (gene_col, gene_label, gene_color) in enumerate(zip(gene_columns, gene_labels, gene_colors)):

            plt.figure(figsize=(14, 10))
            valid_frogs = 0

            for frog_id in frog_ids:

                individual_frog_data = frog_data[frog_data['id'] == frog_id].copy()

                if len(individual_frog_data) == 0:
                    continue

                individual_frog_data = individual_frog_data.sort_values('generation')
                frog_generations = individual_frog_data['generation'].tolist()
                gene_values = individual_frog_data[gene_col].tolist()

                if len(frog_generations) < 2:
                    continue

                valid_data = [(gen, val) for gen, val in zip(frog_generations, gene_values) if pd.notna(val)]
                if len(valid_data) < 2:
                    continue

                valid_generations, valid_gene_values = zip(*valid_data)

                frog_alpha = 0.6
                frog_color = gene_color

                plt.plot(valid_generations, valid_gene_values,
                         color=frog_color, alpha=frog_alpha,
                         linewidth=1.5,
                         linestyle=run_styles[run_id % len(run_styles)],
                         label=f'Frog {frog_id}' if int( frog_id) < 10 else None)

                if len(valid_generations) > 0:
                    plt.scatter(valid_generations[-1], valid_gene_values[-1],
                                color=frog_color, alpha=0.8,
                                s=30, edgecolors='black', linewidth=0.5,
                                zorder=3)

                valid_frogs += 1

            if valid_frogs == 0:
                print(f"No valid frog data found for gene {gene_label} in run {run_id}")
                plt.close()
                continue

            avg_values_per_gen = []
            for gen in generations:
                gen_data = frog_data[frog_data['generation'] == gen]
                if len(gen_data) > 0 and gene_col in gen_data.columns:
                    avg_value = gen_data[gene_col].mean()
                    if pd.notna(avg_value):
                        avg_values_per_gen.append((gen, avg_value))

            if avg_values_per_gen:
                avg_generations, avg_values = zip(*avg_values_per_gen)
                plt.plot(avg_generations, avg_values, color='black', linewidth=4, alpha=0.8,
                         label='Population Average', zorder=5)

                plt.scatter(avg_generations[-1], avg_values[-1], color='black', s=100, marker='X', edgecolors='white',
                            linewidth=2, alpha=1.0, zorder=6)

            plt.xlabel('Generation', fontsize=12, fontweight='bold')
            plt.ylabel(f'{gene_label} Gene Value', fontsize=12, fontweight='bold')
            plt.title(f'Run {run_id}: {gene_label} Evolution - All Individual Frogs', fontsize=14, fontweight='bold', pad=20)

            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

            plt.grid(True, alpha=0.3)

            if valid_frogs <= 15:
                plt.legend(loc='best', framealpha=0.9, fontsize=8, ncol=2)
            else:
                handles, labels = plt.gca().get_legend_handles_labels()
                avg_handle = [h for h, l in zip(handles, labels) if 'Average' in l]
                avg_label = [l for l in labels if 'Average' in l]
                if avg_handle:
                    plt.legend(avg_handle, avg_label, loc='best', framealpha=0.9, fontsize=10)

            all_gene_values = frog_data[gene_col].dropna().tolist()
            if all_gene_values:
                y_min, y_max = min(all_gene_values), max(all_gene_values)
                y_range = y_max - y_min
                if y_range > 0:
                    plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

            if avg_values_per_gen:
                initial_avg = avg_values_per_gen[0][1]
                final_avg = avg_values_per_gen[-1][1]
                change = final_avg - initial_avg

            plt.tight_layout()

            output_path = run_output_dir / f"{gene_col}_all_frogs_evolution.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to: {output_path}")

            plt.close()


def plot_behavior_count_charts(all_run_data, params):

    if isinstance(all_run_data, pd.DataFrame):
        all_run_data = {0: all_run_data}


    gene_columns = ['wall_gene', 'food_gene', 'snake_gene', 'grass_gene', 'water_gene']
    gene_labels = ['Wall Avoidance', 'Food Seeking', 'Snake Avoidance', 'Grass Seeking', 'Water Seeking']

    base_output_dir = Path("plots/consensus/plot_behavior_count_charts")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    for run_id, run_data in all_run_data.items():

        consensus_data = run_data[run_data['species'] == 'consensus'].copy()

        print(f"Processing behavior counts for run {run_id}")

        avg_behavior_counts = {}
        gene_behavior_data = {}

        for gene_col, gene_label in zip(gene_columns, gene_labels):
            gene_consensus = consensus_data[consensus_data['id'] == gene_col]

            if len(gene_consensus) == 0:
                avg_behavior_counts[gene_label] = 0
                gene_behavior_data[gene_label] = []
                continue

            behavior_counts = gene_consensus['flies_eaten_or_behavior_count'].dropna().tolist()

            if len(behavior_counts) > 0:
                avg_count = np.mean(behavior_counts)
                avg_behavior_counts[gene_label] = avg_count
                gene_behavior_data[gene_label] = behavior_counts
                print(f"  {gene_label}: {len(behavior_counts)} records, avg = {avg_count:.1f}")
            else:
                avg_behavior_counts[gene_label] = 0
                gene_behavior_data[gene_label] = []

        plt.figure(figsize=(12, 10))

        pie_labels = []
        pie_values = []
        pie_colors = []

        for i, (gene_label, count) in enumerate(avg_behavior_counts.items()):
            if count > 0:
                pie_labels.append(gene_label)
                pie_values.append(count)
                pie_colors.append(gene_colors[i])

        if len(pie_values) > 0:
            wedges, texts, autotexts = plt.pie(pie_values, labels=pie_labels, colors=pie_colors,autopct='%1.1f%%',
                                               startangle=90, explode=[0.05] * len(pie_values))

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(11)

            for text in texts:
                text.set_fontsize(12)
                text.set_fontweight('bold')

            plt.title(f'Run {run_id}: Average Behavior Count Distribution by Gene', fontsize=16,
                        fontweight='bold', pad=20)

            total_count = sum(pie_values)


            plt.axis('equal')
            plt.tight_layout()

            pie_output_path = base_output_dir / f"run_{run_id}_behavior_counts_pie.png"
            plt.savefig(pie_output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Pie chart saved to: {pie_output_path}")
            plt.close()

        plt.figure(figsize=(12, 8))

        bar_labels = list(avg_behavior_counts.keys())
        bar_values = list(avg_behavior_counts.values())
        bar_colors = gene_colors[:len(bar_labels)]

        bars = plt.bar(bar_labels, bar_values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)

        for bar, value in zip(bars, bar_values):
            if value > 0:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(bar_values) * 0.01,
                         f'{value:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

        plt.xlabel('Gene Type', fontsize=12, fontweight='bold')
        plt.ylabel('Average Behavior Count', fontsize=12, fontweight='bold')
        plt.title(f'Run {run_id}: Average Behavior Count by Gene Type', fontsize=14, fontweight='bold', pad=20)

        plt.xticks(rotation=45, ha='right')

        plt.grid(True, alpha=0.3, axis='y')

        plt.ylim(bottom=0)

        plt.tight_layout()

        bar_output_path = base_output_dir / f"run_{run_id}_behavior_counts_bar.png"
        plt.savefig(bar_output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Bar chart saved to: {bar_output_path}")
        plt.close()


def plot_final_state_stacked_bars(all_run_data, params, show_percentages=True):

    if isinstance(all_run_data, pd.DataFrame):

        all_run_data = {0: all_run_data}

    if len(all_run_data) == 0:
        print("No data loaded from any runs!")
        return

    final_states = ['success', 'alive', 'wall', 'starvation', 'eaten_by_snake']
    state_labels = ['Success', 'Alive', 'Hit Wall', 'Starvation', 'Eaten by Snake']
    state_colors = ['#2ECC71', '#3498DB', '#E74C3C', '#F39C12', '#9B59B6']

    base_output_dir = Path("plots/consensus/plot_final_state_stacked_bars")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    for run_id, run_data in all_run_data.items():

        frog_data = run_data[run_data['species'] == 'frog'].copy()

        print(f"Processing final states for run {run_id}")

        generations = sorted(frog_data['generation'].unique())

        state_counts_per_gen = {}
        total_frogs_per_gen = {}

        for gen in generations:
            gen_data = frog_data[frog_data['generation'] == gen]
            total_frogs = len(gen_data)
            total_frogs_per_gen[gen] = total_frogs

            state_counts = {}
            for state in final_states:
                count = len(gen_data[gen_data['final_state_or_direction'] == state])
                state_counts[state] = count

            state_counts_per_gen[gen] = state_counts

            for state, count in state_counts.items():
                if count > 0:
                    percentage = (count / total_frogs * 100) if total_frogs > 0 else 0
                    print(f"    {state}: {count} ({percentage:.1f}%)")

        plt.figure(figsize=(14, 8))
        bottoms = [0] * len(generations)

        for i, (state, label, color) in enumerate(zip(final_states, state_labels, state_colors)):
            state_counts = [state_counts_per_gen[gen][state] for gen in generations]

            if sum(state_counts) > 0:
                bars = plt.bar(generations, state_counts, bottom=bottoms, label=label, color=color, alpha=0.8,
                               edgecolor='white', linewidth=0.5)

                if show_percentages:
                    for j, (gen, count, bottom) in enumerate(zip(generations, state_counts, bottoms)):
                        if count > 0:
                            total_frogs = total_frogs_per_gen[gen]
                            percentage = (count / total_frogs * 100) if total_frogs > 0 else 0

                            bar_center = bottom + count / 2

                            if percentage >= 3:
                                plt.text(gen, bar_center, f'{percentage:.1f}%', ha='center', va='center', fontsize=9,
                                         fontweight='bold', color='black')

                bottoms = [b + c for b, c in zip(bottoms, state_counts)]

        plt.xlabel('Generation', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Frogs', fontsize=12, fontweight='bold')
        plt.title(f'Run {run_id}: Final State Distribution Across Generations', fontsize=14, fontweight='bold',
                  pad=20)

        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), framealpha=0.9, fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(generations)
        plt.ylim(bottom=0)

        plt.tight_layout()

        output_path = base_output_dir / f"run_{run_id}_final_states_stacked.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Stacked bar chart saved to: {output_path}")
        plt.close()



def plot_gene_behavior_line_graphs(all_run_data, params):

    if isinstance(all_run_data, pd.DataFrame):
        all_run_data = {0: all_run_data}

    gene_columns = ['wall_gene', 'food_gene', 'snake_gene', 'grass_gene', 'water_gene']
    gene_labels = ['Wall Avoidance', 'Food Seeking', 'Snake Avoidance', 'Grass Seeking', 'Water Seeking']

    run_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    run_markers = ['o', 's', '^', 'D', 'v']

    base_output_dir = Path("plots/consensus/plot_gene_behavior_line_graphs")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting gene behavior line graphs")

    for gene_idx, (gene_col, gene_label, gene_color) in enumerate(zip(gene_columns, gene_labels, gene_colors)):

        plt.figure(figsize=(14, 8))

        valid_runs = 0
        all_behavior_values = []

        for run_id, run_data in all_run_data.items():

            consensus_data = run_data[run_data['species'] == 'consensus'].copy()

            gene_consensus = consensus_data[consensus_data['id'] == gene_col]

            generations = sorted(gene_consensus['generation'].unique())
            behavior_counts = []

            for gen in generations:
                gen_data = gene_consensus[gene_consensus['generation'] == gen]
                if len(gen_data) > 0:
                    behavior_count = gen_data['flies_eaten_or_behavior_count'].iloc[0]
                    if pd.notna(behavior_count):
                        behavior_counts.append(behavior_count)
                    else:
                        behavior_counts.append(0)
                else:
                    behavior_counts.append(0)

            plt.plot(generations, behavior_counts, marker=run_markers[run_id % len(run_markers)], linewidth=2.5,
                     markersize=8, color=run_colors[run_id % len(run_colors)],
                     linestyle=run_styles[run_id % len(run_styles)], label=f'Run {run_id}', alpha=0.8)

            if len(generations) > 0 and len(behavior_counts) > 0:
                plt.scatter(generations[-1], behavior_counts[-1], marker='X', s=120,
                            color=run_colors[run_id % len(run_colors)], edgecolors='black', linewidth=2, alpha=1.0,
                            zorder=5)

            all_behavior_values.extend(behavior_counts)
            valid_runs += 1

        plt.xlabel('Generation', fontsize=12, fontweight='bold')
        plt.ylabel(f'{gene_label} Behavior Count', fontsize=12, fontweight='bold')
        plt.title(f'{gene_label}: Behavior Count Evolution Across All Runs', fontsize=14, fontweight='bold', pad=20)

        plt.grid(True, alpha=0.3, axis='y')
        plt.grid(True, alpha=0.2, axis='x')

        plt.legend(loc='best', framealpha=0.9, fontsize=11)

        if all_behavior_values:
            max_value = max(all_behavior_values)
            plt.ylim(bottom=0, top=max_value * 1.1)

        if valid_runs > 0:
            all_generations = []
            for run_data in all_run_data.values():
                if 'species' in run_data.columns:
                    consensus_data = run_data[run_data['species'] == 'consensus']
                    gene_consensus = consensus_data[consensus_data['id'] == gene_col]
                    if len(gene_consensus) > 0:
                        all_generations.extend(gene_consensus['generation'].unique())

            if all_generations:
                unique_generations = sorted(set(all_generations))
                plt.xticks(unique_generations)

        plt.tight_layout()

        output_path = base_output_dir / f"{gene_col}_behavior_evolution_all_runs.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {output_path}")
        plt.close()


def plot_gene_behavior_line_graphs_bestfit(all_run_data, params):

    if isinstance(all_run_data, pd.DataFrame):
        all_run_data = {0: all_run_data}

    gene_columns = ['wall_gene', 'food_gene', 'snake_gene', 'grass_gene', 'water_gene']
    gene_labels = ['Wall Avoidance', 'Food Seeking', 'Snake Avoidance', 'Grass Seeking', 'Water Seeking']

    run_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    run_markers = ['o', 's', '^', 'D', 'v']

    base_output_dir = Path("plots/consensus/plot_gene_behavior_line_graphs")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting gene behavior line graphs")

    for gene_idx, (gene_col, gene_label, gene_color) in enumerate(zip(gene_columns, gene_labels, gene_colors)):

        plt.figure(figsize=(14, 8))

        valid_runs = 0
        all_behavior_values = []
        slope = None
        intercept = None
        r_squared = None
        all_generations_combined = []
        all_behavior_counts_combined = []

        for run_id, run_data in all_run_data.items():

            consensus_data = run_data[run_data['species'] == 'consensus'].copy()
            gene_consensus = consensus_data[consensus_data['id'] == gene_col]

            generations = sorted(gene_consensus['generation'].unique())
            behavior_counts = []

            for gen in generations:
                gen_data = gene_consensus[gene_consensus['generation'] == gen]
                if len(gen_data) > 0:
                    behavior_count = gen_data['flies_eaten_or_behavior_count'].iloc[0]
                    if pd.notna(behavior_count):
                        behavior_counts.append(behavior_count)
                    else:
                        behavior_counts.append(0)
                else:
                    behavior_counts.append(0)

            plt.plot(generations, behavior_counts, marker=run_markers[run_id % len(run_markers)],
                     linewidth=2.5, markersize=8, color=run_colors[run_id % len(run_colors)],
                     linestyle=run_styles[run_id % len(run_styles)], label=f'Run {run_id}', alpha=0.8)

            if len(generations) > 0 and len(behavior_counts) > 0:
                plt.scatter(generations[-1], behavior_counts[-1],
                            marker='X', s=120,
                            color=run_colors[run_id % len(run_colors)],
                            edgecolors='black', linewidth=2,
                            alpha=1.0, zorder=5)

            all_behavior_values.extend(behavior_counts)
            valid_runs += 1

            for gen, count in zip(generations, behavior_counts):
                if pd.notna(count):
                    all_generations_combined.append(gen)
                    all_behavior_counts_combined.append(count)

        all_generations_combined = []
        all_behavior_counts_combined = []

        for run_id, run_data in all_run_data.items():
            if 'species' not in run_data.columns:
                continue

            consensus_data = run_data[run_data['species'] == 'consensus'].copy()
            if len(consensus_data) == 0:
                continue

            gene_consensus = consensus_data[consensus_data['id'] == gene_col]
            if len(gene_consensus) == 0:
                continue

            for gen in sorted(gene_consensus['generation'].unique()):
                gen_data = gene_consensus[gene_consensus['generation'] == gen]
                if len(gen_data) > 0:
                    behavior_count = gen_data['flies_eaten_or_behavior_count'].iloc[0]
                    if pd.notna(behavior_count):
                        all_generations_combined.append(gen)
                        all_behavior_counts_combined.append(behavior_count)

        if len(all_generations_combined) >= 2:
            slope, intercept = np.polyfit(all_generations_combined, all_behavior_counts_combined, 1)

            min_gen = min(all_generations_combined)
            max_gen = max(all_generations_combined)
            fit_generations = np.linspace(min_gen, max_gen, 100)
            fit_values = slope * fit_generations + intercept

            plt.plot(fit_generations, fit_values,
                     color='black', linewidth=3, linestyle='--', alpha=0.7,
                     label=f'Best Fit (slope: {slope:.1f})', zorder=4)

            y_pred = slope * np.array(all_generations_combined) + intercept
            ss_res = np.sum((np.array(all_behavior_counts_combined) - y_pred) ** 2)
            ss_tot = np.sum((np.array(all_behavior_counts_combined) - np.mean(all_behavior_counts_combined)) ** 2)

        if valid_runs == 0:
            print(f"No valid behavior data found for gene {gene_label}")
            plt.close()
            continue

        plt.xlabel('Generation', fontsize=12, fontweight='bold')
        plt.ylabel(f'{gene_label} Behavior Count', fontsize=12, fontweight='bold')
        plt.title(f'{gene_label}: Behavior Count Evolution Across All Runs', fontsize=14, fontweight='bold', pad=20)

        plt.grid(True, alpha=0.3, axis='y')
        plt.grid(True, alpha=0.2, axis='x')

        plt.legend(loc='best', framealpha=0.9, fontsize=11)

        if all_behavior_values:
            max_value = max(all_behavior_values)
            plt.ylim(bottom=0, top=max_value * 1.1)

        if valid_runs > 0:
            all_generations = []
            for run_data in all_run_data.values():
                if 'species' in run_data.columns:
                    consensus_data = run_data[run_data['species'] == 'consensus']
                    gene_consensus = consensus_data[consensus_data['id'] == gene_col]
                    if len(gene_consensus) > 0:
                        all_generations.extend(gene_consensus['generation'].unique())

            if all_generations:
                unique_generations = sorted(set(all_generations))
                plt.xticks(unique_generations)


        plt.tight_layout()

        output_path = base_output_dir / f"{gene_col}_behavior_evolution_all_runs.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {output_path}")
        plt.close()



def main():
    print("Loading consensus evolution data from all ensemble runs")

    df, params, all_run_data = load_and_analyze_data()

    print(f"Data loaded from {len(all_run_data)} runs")

    for run_id, run_df in all_run_data.items():
        print(f"Run {run_id}: {len(run_df)} records, generations {run_df['generation'].min()}-{run_df['generation'].max()}")
        print(f"  Species: {run_df['species'].unique()}")


    create_gene_evolution_plot(all_run_data, params)

    compare_gene_across_runs(all_run_data, params)

    compare_consensus_confidence_across_runs(all_run_data, params)

    plot_individual_frog_gene_evolution(all_run_data, params)

    plot_behavior_count_charts(all_run_data, params)

    plot_final_state_stacked_bars(all_run_data, params, show_percentages=False)

    plot_gene_behavior_line_graphs(all_run_data, params)
    #plot_gene_behavior_line_graphs_bestfit(all_run_data, params)


if __name__ == "__main__":
    main()