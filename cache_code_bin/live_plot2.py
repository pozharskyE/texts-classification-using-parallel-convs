import matplotlib.pyplot as plt


class LivePlot():
    def __init__(self, slice_num: int = 5, slice_frac: float = 0.1, only_train: bool = False):
        self.slice_num = slice_num
        self.slice_frac = slice_frac
        self.only_train = only_train

        if self.only_train:
            self.fig, (self.ax1, self.ax2) = plt.subplots(nrows=2, ncols=1)

            self.line1 = self.ax1.plot([])
            self.line2 = self.ax2.plot([])

            for ax in [self.ax1, self.ax2]:
                ax.set_xlabel('Step')
                ax.set_ylabel('J (cost function value)')

            self.ax1.set_title('J_train')
            self.ax2.set_title('J_train (last X steps)')
            
        
    def update(self, ):
        pass
