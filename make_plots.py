import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


def main():
    for direction, title_str in zip(['dblp_to_acm', 'acm_to_dblp'], ['DBLP -> ACM', 'ACM -> DBLP']):
        for mode in ['SOGA', 'IMOnly', 'SCOnly']:
            val_accs = np.load(f'checkpoints/{direction}_{mode}/acc.npy')
            zero_shot = np.array([float(open(f'checkpoints/{direction}_{mode}/pretrained_results.txt').readlines()[-1].split()[-1])])
            val_accs = np.concatenate([zero_shot, val_accs])
            plt.plot(range(len(val_accs)), val_accs, label=mode)
        plt.legend()
        plt.title(f'Replication Experiment: {title_str}')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.savefig('plots/dblp_acm_replication.svg')

if(__name__ == '__main__'):
    main()