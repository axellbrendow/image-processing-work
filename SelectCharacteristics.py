import tkinter as tk

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
class SelectCharacteristics:
    def __init__(self, parent: tk.Tk) -> None:
        self.parent = parent
        self.root = tk.Toplevel(self.parent)
        self.root.title("Select the characteristics")

        self.energyCheckVar = tk.IntVar(value=1)
        self.energyCheck = tk.Checkbutton(
            self.root,
            text='Energy',
            variable=self.energyCheckVar,
            onvalue=1,
            offvalue=0
        )
        self.energyCheck.pack()

        self.contrastVar = tk.IntVar(value=1)
        self.contrast = tk.Checkbutton(
            self.root,
            text='Contrast',
            variable=self.contrastVar,
            onvalue=1,
            offvalue=0
        )
        self.contrast.pack()

        self.correlationVar = tk.IntVar(value=1)
        self.correlation = tk.Checkbutton(
            self.root,
            text='Correlation',
            variable=self.correlationVar,
            onvalue=1,
            offvalue=0
        )
        self.correlation.pack()

        self.varianceVar = tk.IntVar(value=1)
        self.variance = tk.Checkbutton(
            self.root,
            text='Variance',
            variable=self.varianceVar,
            onvalue=1,
            offvalue=0
        )
        self.variance.pack()

        self.homogeneityVar = tk.IntVar(value=1)
        self.homogeneity = tk.Checkbutton(
            self.root,
            text='Homogeneity',
            variable=self.homogeneityVar,
            onvalue=1,
            offvalue=0
        )
        self.homogeneity.pack()

        self.sumAverageVar = tk.IntVar(value=1)
        self.sumAverage = tk.Checkbutton(
            self.root,
            text='Sum Average',
            variable=self.sumAverageVar,
            onvalue=1,
            offvalue=0
        )
        self.sumAverage.pack()

        self.sumVarianceVar = tk.IntVar(value=1)
        self.sumVariance = tk.Checkbutton(
            self.root,
            text='Sum Variance',
            variable=self.sumVarianceVar,
            onvalue=1,
            offvalue=0
        )
        self.sumVariance.pack()

        self.sumEntropyVar = tk.IntVar(value=1)
        self.sumEntropy = tk.Checkbutton(
            self.root,
            text='Sum Entropy',
            variable=self.sumEntropyVar,
            onvalue=1,
            offvalue=0
        )
        self.sumEntropy.pack()

        self.entropyVar = tk.IntVar(value=1)
        self.entropy = tk.Checkbutton(
            self.root,
            text='Entropy',
            variable=self.entropyVar,
            onvalue=1,
            offvalue=0
        )
        self.entropy.pack()

        self.differenceVarianceVar = tk.IntVar(value=1)
        self.differenceVariance = tk.Checkbutton(
            self.root,
            text='Difference Variance',
            variable=self.differenceVarianceVar,
            onvalue=1,
            offvalue=0
        )
        self.differenceVariance.pack()

        self.differenceEntropyVar = tk.IntVar(value=1)
        self.differenceEntropy = tk.Checkbutton(
            self.root,
            text='Difference Entropy',
            variable=self.differenceEntropyVar,
            onvalue=1,
            offvalue=0
        )
        self.differenceEntropy.pack()

        self.informationMeasuresOfCorrelation12Var = tk.IntVar(value=1)
        self.informationMeasuresOfCorrelation12 = tk.Checkbutton(
            self.root,
            text='Information Measures of Correlation 12',
            variable=self.informationMeasuresOfCorrelation12Var,
            onvalue=1,
            offvalue=0
        )
        self.informationMeasuresOfCorrelation12.pack()

        self.informationMeasuresOfCorrelation13Var = tk.IntVar(value=1)
        self.informationMeasuresOfCorrelation13 = tk.Checkbutton(
            self.root,
            text='Information Measures of Correlation 13',
            variable=self.informationMeasuresOfCorrelation13Var,
            onvalue=1,
            offvalue=0
        )
        self.informationMeasuresOfCorrelation13.pack()

        self.sevenInvariantHuMomentsVar = tk.IntVar(value=1)
        self.sevenInvariantHuMoments = tk.Checkbutton(
            self.root,
            text='7 invariant Hu moments',
            variable=self.sevenInvariantHuMomentsVar,
            onvalue=1,
            offvalue=0
        )
        self.sevenInvariantHuMoments.pack()
