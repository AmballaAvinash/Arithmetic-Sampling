import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def set_matplotlib_font(font_family='normal', font_weight='bold', font_size=17):
    """Set the font properties for matplotlib."""
    font = {'family': font_family,
            'weight': font_weight,
            'size': font_size}
    matplotlib.rc('font', **font)

def combine_means_stds(data_dict):
    """Combine means and standard deviations from all subsets in the data dictionary."""
    all_means = {}
    all_stds = {}
    for subset in data_dict.values():
        for key, value in subset['mean'].items():
            if key not in all_means:
                all_means[key] = []
            all_means[key].append(value)

        for key, value in subset['std'].items():
            if key not in all_stds:
                all_stds[key] = []
            all_stds[key].append(value)

    combined_means = {key: np.mean(values) for key, values in all_means.items()}
    combined_stds = {}

    for key, values in all_stds.items():
        # Calculate combined standard deviation
        n = sum(len(data_dict[subset]['mean'][key]) for subset in data_dict)
        combined_stds[key] = np.sqrt(np.sum([(len(values) - 1) * val ** 2 for val in values]) / (n - len(values)))

    return combined_means, combined_stds

def plot_data(data_dict, label, color, linestyle, num_sampled_sequences, data_range):
    """Plot data from a given data dictionary with specified label, color, and line style."""
    combined_means = data_dict[f'{data_range[0]}-{data_range[1]}']['mean']
    combined_stds = data_dict[f'{data_range[0]}-{data_range[1]}']['std']
    keys = list(combined_means.keys())[:num_sampled_sequences]
    means = [combined_means[key] for key in keys]
    stds = [combined_stds[key] for key in keys]

    plt.errorbar(keys, means, yerr=stds, fmt='o', capsize=5, label=label, color=color, linestyle=linestyle)
    plt.plot(keys, means, color=color, linestyle=linestyle)

    area = np.trapz(means, keys)
    print(f"Area under the {label} curve: {area:.2f}")

def generate_plot(sample_data, arithmetic_data, greedy_data, num_sampled_sequences=40, data_range=(0, 500), title='', xlabel='Number of sampled sequences', ylabel='Accuracy'):
    """Generate and display the plot using the provided data."""
    set_matplotlib_font()

    fig, ax = plt.subplots(figsize=(10, 6))

    plot_data(sample_data, 'Ancestral', 'blue', '--', num_sampled_sequences, data_range)
    plot_data(greedy_data, 'Greedy', 'black', ':', num_sampled_sequences, data_range)
    plot_data(arithmetic_data, 'Arithmetic', 'red', '-', num_sampled_sequences, data_range)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=17)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.show()

# Example usage
sample_data = {
    '0-1221':{'mean': {1: 0.4030958230958231,
  2: 0.44807534807534793,
  4: 0.5240786240786242,
  5: 0.5466502866502867,
  8: 0.5785749385749385,
  10: 0.590925470925471,
  20: 0.6178050778050779,
  40: 0.6265356265356266},
 'std': {1: 0.011489025692324425,
  2: 0.010950442173458008,
  4: 0.009607476558239392,
  5: 0.01025447897559759,
  8: 0.008802883537332508,
  10: 0.009332642805869958,
  20: 0.007282404965064158,
  40: 0}}}

# Arithmetic data
arithmetic_data ={
    '0-1221': {'mean': {1: 0.40606060606060607,
  2: 0.45606060606060606,
  4: 0.5311220311220313,
  5: 0.5483210483210483,
  8: 0.5791154791154792,
  10: 0.589025389025389,
  20: 0.6151924651924652,
  40: 0.6314496314496314},
 'std': {1: 0.014969130496108032,
  2: 0.013211767755638907,
  4: 0.008386264454975924,
  5: 0.008265418590986225,
  8: 0.00908648098100637,
  10: 0.008200648289080419,
  20: 0.005023854098766688,
  40: 0}}}

greedy_data = {
    '0-1221': {'mean': {1: 0.391,
  2: 0.391,
  4: 0.391,
  5: 0.391,
  8: 0.391,
  10: 0.391,
  20: 0.391,
  40: 0.391},
 'std': {1: 0,
  2: 0,
  4: 0,
  5: 0,
  8: 0,
  10: 0,
  20: 0,
  40: 0}}}
generate_plot(sample_data, arithmetic_data, greedy_data,
              num_sampled_sequences=40, data_range=(0, 1221),
              title='Llama-2-7b,Commonsense QA,t=1,top-p=0.95,dataset size=1221')