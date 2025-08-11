import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

run_styles = ['-', '-', '-', '-', '-']
gene_colors = ['#E74C3C', '#8B4513', '#FF8C00', '#2ECC71', '#3498DB']


def load_data():

    base_path = Path("training data/GA/")
    selection_methods = ['tournament', 'steady', 'GA-SA']

    GA_type_files = {}
    params = {}

    for method in selection_methods:
        csv_path = base_path / f"{method}_agent_data.csv"
        if csv_path.exists():
            GA_type_files[method] = pd.read_csv(csv_path)
            print(f"Loaded: {csv_path}")

        json_path = base_path / f"{method}_params.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                params[method] = json.load(f)
            print(f"Loaded: {json_path}")

    return GA_type_files, params


def plot_fitness_evolution(GA_type_files, params):

    output_dir = Path("plots/GA/plot_fitness_evolution")
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = ['tournament', 'steady', 'GA-SA']

    for method in methods:
        if method in GA_type_files:
            print(f"Starting fitness evolution for {method}")

            plt.figure(figsize=(12, 8))

            frog_data = GA_type_files[method][GA_type_files[method]['species'] == 'frog']
            stats = frog_data.groupby('generation')['fitness'].agg([
                ('best', 'max'),
                ('average', 'mean'),
                ('worst', 'min')
            ]).reset_index()


            plt.plot(stats['generation'], stats['best'], color=gene_colors[3], linewidth=2.5, linestyle='-',
                     marker='o', markersize=6, label='Best', alpha=0.8)

            plt.plot(stats['generation'], stats['average'], color=gene_colors[4], linewidth=2.5, linestyle='-',
                     marker='s', markersize=6, label='Average', alpha=0.8)

            plt.plot(stats['generation'], stats['worst'], color=gene_colors[0], linewidth=2.5, linestyle='-',
                     marker='^', markersize=6, label='Worst', alpha=0.8)

            plt.title(f'{method.upper()} Selection Method - Fitness Evolution', fontsize=14, fontweight='bold', pad=20)
            plt.xlabel('Generation', fontsize=12, fontweight='bold')
            plt.ylabel('Fitness', fontsize=12, fontweight='bold')
            plt.legend(loc='best', framealpha=0.9, fontsize=11)
            plt.grid(True, alpha=0.3)

            plt.xticks(stats['generation'].unique())

            plt.tight_layout()

            output_path = output_dir / f'{method}_fitness_evolution.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to: {output_path}")

            plt.close()


def plot_final_state_stacked_bars(data_files, param_files, show_percentages=True):

    final_states = ['success', 'alive', 'wall', 'starvation', 'eaten_by_snake']
    state_labels = ['Success', 'Alive', 'Hit Wall', 'Starvation', 'Eaten by Snake']
    state_colors = ['#2ECC71', '#3498DB', '#E74C3C', '#F39C12', '#9B59B6']

    output_dir = Path("plots/GA/plot_final_state_stacked_bars")
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = ['tournament', 'steady', 'GA-SA']

    for method in methods:

        frog_data = data_files[method][data_files[method]['species'] == 'frog'].copy()

        generations = sorted(frog_data['generation'].unique())

        state_counts_per_gen = {}
        total_frogs_per_gen = {}

        for gen in generations:
            gen_data = frog_data[frog_data['generation'] == gen]
            total_frogs = len(gen_data)
            total_frogs_per_gen[gen] = total_frogs

            state_counts = {}
            for state in final_states:
                count = len(gen_data[gen_data['final_state'] == state])
                state_counts[state] = count

            state_counts_per_gen[gen] = state_counts

            print(f"Generation {gen}: {total_frogs} frogs")
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

                            if percentage >= 5:
                                plt.text(gen, bottom + count / 2, f'{percentage:.0f}%', ha='center', va='center',
                                         fontweight='bold', fontsize=9, color='black')

                bottoms = [b + c for b, c in zip(bottoms, state_counts)]

        plt.xlabel('Generation', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Frogs', fontsize=12, fontweight='bold')
        plt.title(f'{method.upper()}: Final State Distribution Across Generations', fontsize=14, fontweight='bold',
                  pad=20)

        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), framealpha=0.9, fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(generations)
        plt.ylim(bottom=0)
        plt.tight_layout()

        output_path = output_dir / f"{method}_final_states_stacked.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Stacked bar chart saved to: {output_path}")
        plt.close()


def plot_gene_evolution_individual(data_files, param_files):

    output_dir = Path("plots/GA/plot_gene_evolution_individual")
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = ['tournament', 'steady', 'GA-SA']
    gene_columns = ['wall_gene', 'food_gene', 'snake_gene', 'grass_gene', 'water_gene']
    gene_labels = ['Wall Avoidance', 'Food Seeking', 'Snake Avoidance', 'Grass Seeking', 'Water Seeking']

    for method in methods:

        frog_data = data_files[method][data_files[method]['species'] == 'frog'].copy()

        generations = sorted(frog_data['generation'].unique())

        plt.figure(figsize=(12, 8))

        valid_genes = 0

        for i, (gene_col, gene_label) in enumerate(zip(gene_columns, gene_labels)):

            avg_values = []
            for gen in generations:
                gen_data = frog_data[frog_data['generation'] == gen]
                avg_value = gen_data[gene_col].mean()
                avg_values.append(avg_value)

            if not avg_values or all(pd.isna(val) for val in avg_values):
                continue

            plt.plot(generations, avg_values, color=gene_colors[i % len(gene_colors)], linewidth=2.5, linestyle='-',
                     marker='o', markersize=6, label=gene_label, alpha=0.8)

            valid_genes += 1


        plt.xlabel('Generation', fontsize=12, fontweight='bold')
        plt.ylabel('Average Gene Value', fontsize=12, fontweight='bold')
        plt.title(f'{method.upper()}: Gene Evolution Over Time', fontsize=14, fontweight='bold', pad=20)

        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        plt.grid(True, alpha=0.3)
        plt.ylim(-1.0, 1.0)
        plt.legend(loc='best', framealpha=0.9, fontsize=10)

        plt.xticks(generations)
        plt.tight_layout()

        output_path = output_dir / f"{method}_gene_evolution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {output_path}")
        plt.close()


def plot_gene_evolution_comparison(data_files, param_files):

    output_dir = Path("plots/GA/plot_gene_evolution_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = ['tournament', 'steady', 'GA-SA']
    method_colors = ['#E74C3C', '#2ECC71', '#3498DB']
    method_markers = ['o', 's', '^']

    gene_columns = ['wall_gene', 'food_gene', 'snake_gene', 'grass_gene', 'water_gene']
    gene_labels = ['Wall Avoidance', 'Food Seeking', 'Snake Avoidance', 'Grass Seeking', 'Water Seeking']

    for gene_idx, (gene_col, gene_label) in enumerate(zip(gene_columns, gene_labels)):

        plt.figure(figsize=(12, 8))
        valid_methods = 0

        for i, method in enumerate(methods):
            if method not in data_files:
                continue

            frog_data = data_files[method][data_files[method]['species'] == 'frog'].copy()

            generations = sorted(frog_data['generation'].unique())
            avg_values = []

            for gen in generations:
                gen_data = frog_data[frog_data['generation'] == gen]
                avg_value = gen_data[gene_col].mean()
                avg_values.append(avg_value)

            if not avg_values or all(pd.isna(val) for val in avg_values):
                continue

            plt.plot(generations, avg_values,
                     color=method_colors[i % len(method_colors)], linewidth=2.5, linestyle='-',
                     marker=method_markers[i % len(method_markers)], markersize=6, label=f'{method.upper()}', alpha=0.8)

            if len(generations) > 0 and len(avg_values) > 0:
                plt.scatter(generations[-1], avg_values[-1],
                            marker='X', s=120,
                            color=method_colors[i % len(method_colors)],
                            edgecolors='black', linewidth=2,
                            alpha=1.0, zorder=5)

            valid_methods += 1

        plt.xlabel('Generation', fontsize=12, fontweight='bold')
        plt.ylabel(f'{gene_label} Gene Value', fontsize=12, fontweight='bold')
        plt.title(f'{gene_label}: Evolution Comparison Across Selection Methods', fontsize=14, fontweight='bold', pad=20)

        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        plt.grid(True, alpha=0.3)
        plt.ylim(-1.0, 1.0)
        plt.legend(loc='best', framealpha=0.9, fontsize=10)

        plt.tight_layout()

        output_path = output_dir / f"{gene_col}_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {output_path}")
        plt.close()


def plot_individual_frog_gene_evolution(data_files, param_files):

    base_output_dir = Path("plots/GA/plot_individual_frog_gene_evolution")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    methods = ['tournament', 'steady', 'GA-SA']
    gene_columns = ['wall_gene', 'food_gene', 'snake_gene', 'grass_gene', 'water_gene']
    gene_labels = ['Wall Avoidance', 'Food Seeking', 'Snake Avoidance', 'Grass Seeking', 'Water Seeking']

    for method in methods:

        frog_data = data_files[method][data_files[method]['species'] == 'frog'].copy()

        method_output_dir = base_output_dir / method
        method_output_dir.mkdir(parents=True, exist_ok=True)

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

                plt.plot(valid_generations, valid_gene_values, color=frog_color, alpha=frog_alpha, linewidth=1.5,
                         linestyle='-', label=f'Frog {frog_id}' if int(frog_id) < 10 else None)

                if len(valid_generations) > 0:
                    plt.scatter(valid_generations[-1], valid_gene_values[-1], color=frog_color, alpha=0.8,
                                s=30, edgecolors='black', linewidth=0.5, zorder=3)

                valid_frogs += 1


            avg_values_per_gen = []
            for gen in generations:
                gen_data = frog_data[frog_data['generation'] == gen]
                if len(gen_data) > 0 and gene_col in gen_data.columns:
                    avg_value = gen_data[gene_col].mean()
                    if pd.notna(avg_value):
                        avg_values_per_gen.append((gen, avg_value))

            if avg_values_per_gen:
                avg_generations, avg_values = zip(*avg_values_per_gen)
                plt.plot(avg_generations, avg_values, color='black', linewidth=4, alpha=0.8, label='Population Average',
                         zorder=5)

                plt.scatter(avg_generations[-1], avg_values[-1], color='black', s=100, marker='X', edgecolors='white',
                            linewidth=2, alpha=1.0, zorder=6)

            plt.xlabel('Generation', fontsize=12, fontweight='bold')
            plt.ylabel(f'{gene_label} Gene Value', fontsize=12, fontweight='bold')
            plt.title(f'{method.upper()}: {gene_label} Evolution - All Individual Frogs', fontsize=14, fontweight='bold',
                      pad=20)

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

            plt.tight_layout()

            output_path = method_output_dir / f"{gene_col}_all_frogs_evolution.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to: {output_path}")

            plt.close()


def main():

    GA_type_files, params = load_data()

    print(f"Data loaded successfully from {len(GA_type_files)}")

    for method, df in GA_type_files.items():
        print(f"{method}: {len(df)} records, generations {df['generation'].min()}-{df['generation'].max()}")
        print(f"  Species: {df['species'].unique()}")

    plot_fitness_evolution(GA_type_files, params)

    plot_final_state_stacked_bars(GA_type_files, params, show_percentages=False)

    plot_gene_evolution_individual(GA_type_files, params)

    plot_gene_evolution_comparison(GA_type_files, params)

    plot_individual_frog_gene_evolution(GA_type_files, params)



if __name__ == "__main__":
    main()