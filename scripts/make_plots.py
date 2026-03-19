import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    os.makedirs("results", exist_ok=True)
    summary_lines = []

    for direction, title_str in zip(['dblp_to_acm', 'acm_to_dblp'], ['DBLP -> ACM', 'ACM -> DBLP']):
        for mode in ['SOGA', 'IMOnly', 'SCOnly']:
            val_accs = np.load(f'checkpoints/{direction}_{mode}/acc.npy')
            zero_shot = np.array([
                float(open(
                    f'checkpoints/{direction}_{mode}/pretrained_results.txt'
                ).readlines()[-1].split()[-1])
            ])

            val_accs = np.concatenate([zero_shot, val_accs])
            best_acc = np.max(val_accs)
            summary_lines.append(f"{direction} | {mode} | best_acc: {best_acc:.6f}")
            plt.plot(range(len(val_accs)), val_accs, label=mode)
        plt.legend()
        plt.title(f'Replication Experiment: {title_str}')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.savefig(f'results/{direction}_replication.svg')
        plt.close()

    with open("results/best_accuracies.txt", "w") as f:
        for line in summary_lines:
            f.write(line + "\n")


if __name__ == '__main__':
    main()