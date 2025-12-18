import matplotlib.pyplot as plt
import seaborn as sns


def create_plots(G, RESULTS_DIR, E=1):
    """
        Plot the distribution of estimated FRW hyperparameters.

        Parameters
        ----------
        G : np.ndarray
            Estimated hyperparameter values across Monte Carlo replications.
        E : int, default=1
            Experiment index used for labeling and saving the figure.
    """
    plt.hist(G, bins=50, density=True, alpha=0.3, color='steelblue', edgecolor='black')
    sns.kdeplot(data=G, color='red', lw=1)
    plt.title('Experiment ' + str(E), fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('')
    plt.savefig(RESULTS_DIR / f"E{E}.pdf")
    plt.show()
