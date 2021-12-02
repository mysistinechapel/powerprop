import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style('white')
sns.set_context('paper')

seed = 0
torch.manual_seed(seed)
rng = np.random.default_rng(seed)


def load_weights(fname):
    return np.load(fname, allow_pickle=True)


# weight distribution
def prune_weights(initial, final, p=0.1):
    cond = np.abs(final.flatten())
    n = int(p * cond.shape[0])
    top_k = np.argsort(cond)[::-1][:n]
    # how_many = int(p * cond.shape[0])
    # top_k = torch.topk(condition, k=how_many).indices

    # remaining weights
    init_rem = initial.flatten()[top_k]
    final_rem = final.flatten()[top_k]
    remaining = np.column_stack((init_rem, final_rem))

    # pruned weights
    mask = np.ones(cond.shape, bool)
    mask[top_k] = False
    init_pruned = initial.flatten()[mask]
    final_pruned = final.flatten()[mask]
    pruned = np.column_stack((init_pruned, final_pruned))

    return remaining, pruned


def sample_weights(w):
    # sample 10k elements
    weights = np.vstack(w)
    idx = rng.choice(np.arange(weights.shape[0]), size=10000)
    return weights[idx, :]


def plot_weights(remaining, pruned):
    n = pruned.shape[0]
    data = {
        'Initial weight': np.hstack((pruned[:, 0], remaining[:, 0])),
        'Final weight': np.hstack((pruned[:, 1], remaining[:, 1])),
        'type': ['Pruned weights'] * n + ['Remaining weights'] * n
    }

    g = sns.jointplot(data=data, x='Initial weight', y='Final weight', hue='type')
    g.ax_joint.legend(frameon=False).set_title(None)
    g.ax_joint.set_yscale('symlog')
    g.ax_joint.set_yticks([-1e1, -1e0, 0.0, 1e0, 1e1])
    g.ax_joint.set_ylim([-2e1, 2e1])
    sns.rugplot(data=data, y='Final weight', hue='type', height=0.5, ax=g.ax_marg_y, legend=False)


def plot_weight_distribution(final, initial, rep=0, alpha=1):
    remains, prunes = [], []
    for fw, iw in zip(final[rep, alpha - 1], initial[rep, alpha - 1]):
        rem, pru = prune_weights(iw, fw)
        remains.append(rem)
        prunes.append(pru)

    remaining = sample_weights(remains)
    pruned = sample_weights(prunes)
    plot_weights(remaining, pruned)

# mask overlap
def create_mask(arr, prob):
    cond = np.abs(arr.flatten())
    return np.argsort(cond)[:int(prob * cond.shape[0])]


def mask_overlap(initial, final, prob):
    init_mask = create_mask(initial, prob)
    final_mask = create_mask(final, prob)
    overlap = np.intersect1d(init_mask, final_mask, assume_unique=True).shape[0]
    return overlap, init_mask.shape[0]


def overlap_by_reps_and_alphas(final, initial):
    reps = final.shape[0]
    alphas = final.shape[1] - 1
    results = []
    for r in range(reps):
        rep_res = []
        for alpha in range(alphas):
            alpha_res = []
            for p in (0.5, 0.85, 0.95):
                total_overlap, total = 0, 0
                for fw, iw in zip(final[r, alpha], initial[r, alpha]):
                    overlap, number = mask_overlap(iw, fw, p)
                    total_overlap += overlap
                    total += number

                alpha_res.append(total_overlap / total)
            rep_res.append(alpha_res)
        results.append(rep_res)

    return np.array(results)


def plot_mask_overlaps(final, initial):
    data = overlap_by_reps_and_alphas(final, initial)
    means = data.mean(axis=0)
    stds = data.std(axis=0)

    f, ax = plt.subplots(1, 1, figsize=(7, 5))

    labels = ['50%', '85%', '95%']
    for i in range(3):
        ax.errorbar(x=[1, 2, 3, 4], y=100 * means[:, i], yerr=100 * stds[:, i], label=labels[i])

    ax.set_xlabel(r'Powerprop. $\alpha$')
    ax.set_ylabel('Mask overlap before/after training (%)')
    ax.set_ylim([70, 100])
    ax.set_xticks([1.0, 2.0, 3.0, 4.0])
    ax.legend(title='Pruning fraction', frameon=False)
    sns.despine()
    plt.show()


# 10k smallest weights
def plot_smallest_weights(weights, rep=0, n=10000):
    res = []
    alphas = np.arange(3)
    for alpha in alphas:
        fws = [fw.flatten() for fw in weights[rep, alpha]]
        fws = np.concatenate(fws)
        smallest = np.argsort(np.abs(fws))[:n]
        res.append(fws[smallest])

    data = {
        'weights': np.hstack(res),
        r'$\alpha$': np.repeat(1.0 + alphas, n)
    }
    f, ax = plt.subplots(1, 1, figsize=(7, 5))
    sns.kdeplot(data=data, x='weights', hue=r'$\alpha$', ax=ax, fill=True)
    ax.set_xlabel('Final weights')
    sns.despine()
    plt.show()


if __name__ == '__main__':
    initial_weights = load_weights(r'../notebooks/data/mnist_initial_weights_lr01_m0_FINAL.npy')
    final_weights = load_weights(r'../notebooks/data/mnist_final_weights_lr01_m0_FINAL.npy')

    # mask overlap plots
    plot_mask_overlaps(final_weights, initial_weights)

    # weight distribution plots
    plot_weight_distribution(final_weights, initial_weights)
    plot_weight_distribution(final_weights, initial_weights, alpha=4)

    # n smallest weights plot
    plot_smallest_weights(final_weights)
