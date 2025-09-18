import torch
import matplotlib.pyplot as plt
from numpy import ndarray, concatenate as npyconcatenate, arange as npyarange
from pandas import DataFrame
import seaborn as sns

def layers_histogram(reference: dict[str, torch.nn.Module], compare: dict[str, torch.nn.Module]):
    layer_dims: list[int] = []
    layer_mses: list[float] = []
    layer_type: list[str] = []
    ldx = 0
    for k in reference.keys():
        if "bias" in k: continue
        if "weight" not in k and "running_mean" not in k and "running_var" not in k: continue
        ref: ndarray = reference[k].flatten().numpy(); cmp: ndarray = compare[k].flatten().numpy()
        hasbias = False
        if "weight" in k and k.replace("weight", "bias") in reference:
            hasbias = True
            refb = reference[k.replace("weight", "bias")].flatten().numpy()
            cmpb = compare[k.replace("weight", "bias")].flatten().numpy()
            ref = npyconcatenate((ref, refb)); cmp = npyconcatenate((cmp, cmpb))    
        indexes = npyarange(ref.shape[0])
        sqerr = (ref - cmp)**2
        layer_dims.append(sqerr.shape[0])
        layer_mses.append(sqerr.mean())
        layer_type.append('conv' if 'conv' in k else 'bn' if 'bn' in k else 'dws' if 'downsample' in k else 'fc')
        fig = plt.figure(figsize=(12,12))
        fig.suptitle(f"Layer {ldx} - {k}{'.bias' if hasbias else ''}", fontsize=12)
        r1c1 = fig.add_subplot(3,2,1)
        r1c1.hist(ref, bins=64, edgecolor="black", color='lightgreen', label='original')
        r1c1.set_title('original')
        r1c2 = fig.add_subplot(3,2,2)
        r1c2.hist(cmp, bins=64, edgecolor='black', color='lightcoral', label='reconstructed')
        r1c2.set_title('reconstructed')
        r2c12 = fig.add_subplot(3,1,2)
        r2c12.boxplot(
            sqerr, patch_artist=True, vert=False,
            boxprops=dict(facecolor='lightblue', edgecolor='blue'),
            whiskerprops=dict(color='gray'),
            capprops=dict(color='gray'),
            medianprops=dict(color='red')
        )
        r2c12.set_xlabel("Squared Error")
        r2c12.set_yticks([])
        r3c12 = fig.add_subplot(3,1,3)
        r3c12.plot(indexes, ref, color='lightgreen', label='original', linewidth=0.8)
        r3c12.plot(indexes, cmp, color='lightcoral', label='reconstructed', linewidth=0.8, alpha=0.5)
        r3c12.set_xlabel("indexes"); r3c12.set_ylabel("values")
        r3c12.legend()
        plt.tight_layout()
        yield ldx, f"{k}.bias" if hasbias else k, fig, sqerr.mean()
        plt.close(fig)
        ldx+=1
    
    scatter = plt.figure(figsize=(10,6))
    scatter_ax = scatter.add_subplot(1,1,1)
    scatter_ax.scatter(layer_dims, layer_mses, alpha=0.7, color='purple', s=50)
    scatter_ax.set_xscale('log'); scatter_ax.set_yscale('log')
    scatter_ax.set_xlabel('dimensionality'); scatter_ax.set_ylabel('MSE')
    plt.grid(True, which="both", ls="--", c='0.7')
    yield -1, "test-plots", scatter, None
    plt.close(scatter)
    df = DataFrame({'type': layer_type, 'mses': layer_mses})
    boxplot = plt.figure(figsize=(10,6))
    boxplot_ax = boxplot.add_subplot(1,1,1)
    sns.boxplot(x='type', y='mses', data=df, ax=boxplot_ax)
    boxplot_ax.set_title("MSE Distribution By Layer")
    boxplot_ax.set_xlabel("Layer Type")
    boxplot_ax.set_ylabel("MSEs")
    yield -1, "test-plots", boxplot, None
    plt.close(boxplot)

def embedding_2dvisualizer(Xdata, Ydata, dataset_names: list[str]):
    scatter = plt.figure(figsize=(12,12))
    scatter_ax = scatter.add_subplot(1,1,1)
    plot = scatter_ax.scatter(Xdata[:, 0], Xdata[:, 1], c=Ydata, s=20, cmap='tab10', alpha=0.8)
    handles, labels = plot.legend_elements()
    legend_labels = [f"{lbl}" for lbl in labels]
    scatter_ax.legend(handles, legend_labels, title=";".join([f"{dsname} = {cdx}" for cdx,dsname in enumerate(dataset_names)]))
    plt.tight_layout()
    yield scatter
    plt.close(scatter)