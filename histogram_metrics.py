import numpy as np
from scipy.spatial import distance
from scipy.stats import pearsonr, spearmanr, entropy
from scipy.spatial.distance import jensenshannon

class ExtendedQuantileMetrics:
    def __init__(self, hist1, bins1, hist2, bins2):
        """
        Initialize the class with two histograms and their respective bin edges.
        hist1, hist2: Arrays representing the histogram counts.
        bins1, bins2: Arrays representing the bin edges.
        """
        self.hist1 = np.array(hist1, dtype=float)
        self.hist2 = np.array(hist2, dtype=float)
        self.bins1 = np.array(bins1, dtype=float)
        self.bins2 = np.array(bins2, dtype=float)

        # Create common bins
        self.common_bins = np.union1d(self.bins1, self.bins2)

        # Rebin the histograms to the common bins
        self.rebinned_hist1 = self.rebin_histogram(self.hist1, self.bins1, self.common_bins)
        self.rebinned_hist2 = self.rebin_histogram(self.hist2, self.bins2, self.common_bins)

    def rebin_histogram(self, hist, original_bins, common_bins):
        """
        Rebin the histogram to a common set of bins.
        """
        # Create an empty array for the new bins
        rebinned = np.zeros(len(common_bins) - 1, dtype=float)
        
        # Fill the new bins with appropriate counts
        for i in range(len(original_bins) - 1):
            # Find the indices where the original bins fall into the common bins
            start = np.searchsorted(common_bins, original_bins[i], side='right') - 1
            end = np.searchsorted(common_bins, original_bins[i + 1], side='left')

            # Distribute the count across the corresponding new bins
            if start < end:
                rebinned[start:end] += hist[i]

        return rebinned

    def euclidean_distance(self):
        """
        Compute the Euclidean distance between rebinned histograms.
        """
        return distance.euclidean(self.rebinned_hist1, self.rebinned_hist2)

    def manhattan_distance(self):
        """
        Compute the Manhattan distance between rebinned histograms.
        """
        return distance.cityblock(self.rebinned_hist1, self.rebinned_hist2)

    def cosine_similarity(self):
        """
        Compute the cosine similarity between rebinned histograms.
        """
        return 1 - distance.cosine(self.rebinned_hist1, self.rebinned_hist2)

    def pearson_correlation(self):
        """
        Compute the Pearson correlation coefficient between rebinned histograms.
        """
        return pearsonr(self.rebinned_hist1, self.rebinned_hist2)[0]

    def jensen_shannon_divergence(self):
        """
        Compute the Jensen-Shannon divergence.
        """
        return jensenshannon(self.rebinned_hist1, self.rebinned_hist2)

