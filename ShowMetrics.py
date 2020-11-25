import tkinter as tk

class ShowMetrics:
    def __init__(
        self,
        parent: tk.Tk,
        executionTime,
        FP,
        FN,
        TP,
        TN,
        TPR,  # Sensitivity, hit rate, recall, or true positive rate
        TNR,  # Specificity or true negative rate
        ACC,  # Overall accuracy
        accuracy
    ) -> None:
        self.parent = parent
        self.root = tk.Toplevel(self.parent)
        self.root.title("Metrics")

        self.metricsLabel = tk.Text(self.root, borderwidth=0)
        self.metricsLabel.insert(1.0, f'''Execution time: {round(executionTime, 3)} sec

Accuracy: {round(accuracy, 2)}

Overall Accuracy:
{[round(x, 2) for x in ACC]} -> {round(ACC.mean(), 2)}

Sensibility:
{[round(x, 2) for x in TPR]} -> {round(TPR.mean(), 2)}

Specificity:
{[round(x, 2) for x in TNR]} -> {round(TNR.mean(), 2)}
''')
        self.metricsLabel.pack(fill='both', expand=True)
