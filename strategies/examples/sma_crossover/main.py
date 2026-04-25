from AlgorithmImports import *


class SmaCrossover(QCAlgorithm):
    """
    Simple 50/200-day SMA crossover on SPY.
    Buy when fast SMA crosses above slow SMA, sell when it crosses below.
    """

    def Initialize(self):
        self.SetStartDate(2018, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(50000)

        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol

        self.fast = self.SMA(self.spy, 50, Resolution.Daily)
        self.slow = self.SMA(self.spy, 200, Resolution.Daily)

    def OnData(self, data):
        if not self.fast.IsReady or not self.slow.IsReady:
            return

        if self.fast.Current.Value > self.slow.Current.Value:
            if not self.Portfolio.Invested:
                self.SetHoldings(self.spy, 1.0)
        else:
            if self.Portfolio.Invested:
                self.Liquidate(self.spy)
