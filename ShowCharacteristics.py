import tkinter as tk

from typing import List

"""
Angular Second Moment: Energy or Uniformity
Contrast
Correlation
Sum of Squares: Variance
Inverse Difference Moment: Texture Homogeneity
Sum Average
Sum Variance
Sum Entropy
Entropy
Difference Variance
Difference Entropy
Information Measures of Correlation
Information Measures of Correlation
7 invariant Hu moments
"""
class ShowCharacteristics:
    def __init__(
        self,
        parent: tk.Tk,
        executionTime,
        energy,
        contrast,
        correlation,
        variance,
        homogeneity,
        sumAverage,
        sumVariance,
        sumEntropy,
        entropy,
        differenceVariance,
        differenceEntropy,
        informationMeasuresOfCorrelation12,
        informationMeasuresOfCorrelation13,
        sevenInvariantHuMoments: List[float]
    ) -> None:
        self.parent = parent
        self.root = tk.Toplevel(self.parent)
        self.root.title("Characteristics")

        self.executionTimeLabel = tk.Label(
            self.root, text=f"Execution time: {executionTime} sec")
        self.executionTimeLabel.pack()

        self.energyLabel = tk.Label(self.root, text=f"Energy: {energy}")
        self.energyLabel.pack()

        self.contrast = tk.Label(self.root, text=f'Contrast: {contrast}')
        self.contrast.pack()

        self.correlation = tk.Label(self.root, text=f'Correlation: {correlation}')
        self.correlation.pack()

        self.variance = tk.Label(self.root, text=f'Variance: {variance}')
        self.variance.pack()

        self.homogeneity = tk.Label(self.root, text=f'Homogeneity: {homogeneity}')
        self.homogeneity.pack()

        self.sumAverage = tk.Label(self.root, text=f'Sum Average: {sumAverage}')
        self.sumAverage.pack()

        self.sumVariance = tk.Label(self.root, text=f'Sum Variance: {sumVariance}')
        self.sumVariance.pack()

        self.sumEntropy = tk.Label(self.root, text=f'Sum Entropy: {sumEntropy}')
        self.sumEntropy.pack()

        self.entropy = tk.Label(self.root, text=f'Entropy: {entropy}')
        self.entropy.pack()

        self.differenceVariance = tk.Label(self.root, 
            text=f'Difference Variance: {differenceVariance}')
        self.differenceVariance.pack()

        self.differenceEntropy = tk.Label(self.root, 
            text=f'Difference Entropy: {differenceEntropy}')
        self.differenceEntropy.pack()

        self.informationMeasuresOfCorrelation12 = tk.Label(self.root, 
            text=f'Information Measures of Correlation 12: {informationMeasuresOfCorrelation12}')
        self.informationMeasuresOfCorrelation12.pack()

        self.informationMeasuresOfCorrelation13 = tk.Label(self.root, 
            text=f'Information Measures of Correlation 13: {informationMeasuresOfCorrelation13}')
        self.informationMeasuresOfCorrelation13.pack()

        for i in range(len(sevenInvariantHuMoments)):
            tk.Label(self.root, 
                text=f"Invariant Hu moment {i + 1}: {sevenInvariantHuMoments[i]}"
            ).pack()
