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

        self.executionTimeLabel = tk.Text(self.root, borderwidth=0)
        self.executionTimeLabel.insert(1.0, f'''Execution time: {round(executionTime, 3)} sec
Energy: {energy}
Contrast: {contrast}
Correlation: {correlation}
Variance: {variance}
Homogeneity: {homogeneity}
Sum Average: {sumAverage}
Sum Variance: {sumVariance}
Sum Entropy: {sumEntropy}
Entropy: {entropy}
Difference Variance: {differenceVariance}
Difference Entropy: {differenceEntropy}
Information Measures of Correlation 12: {informationMeasuresOfCorrelation12}
Information Measures of Correlation 13: {informationMeasuresOfCorrelation13}
Invariant Hu moment 1: {sevenInvariantHuMoments[0]}
Invariant Hu moment 2: {sevenInvariantHuMoments[1]}
Invariant Hu moment 3: {sevenInvariantHuMoments[2]}
Invariant Hu moment 4: {sevenInvariantHuMoments[3]}
Invariant Hu moment 5: {sevenInvariantHuMoments[4]}
Invariant Hu moment 6: {sevenInvariantHuMoments[5]}
Invariant Hu moment 7: {sevenInvariantHuMoments[6]}
''')
        self.executionTimeLabel.pack(fill='both', expand=True)
