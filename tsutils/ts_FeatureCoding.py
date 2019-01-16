class Feature_Coding:
    def __init__(self):
        self.featuresAll = ['TimeStamp', 'Symbol', 'Exchange', 'Type', 'PeriodCode', 'EventId', 'EventCode', 'EventDir',
                            'Price', 'VWAPP',
                            'Open', 'High', 'Low', 'Close', 'Volume',
                            'PIR5m', 'PIR15m', 'PIR60m', 'PIRD', 'PIRYD', 'PIR5D', 'PIR10D', 'PIR20D',
                            'MarketTrend_D', 'MarketTrend_60', 'MarketTrend_15', 'MarketTrend_5', 'MarketTrend_1']
        self.featuresSel = ['TimeStamp', 'Type', 'TypeCode', 'Quote', 'Period',
                            'MarketTrend_D', 'MarketTrend_60', 'MarketTrend_15',
                            'MarketTrend_5', 'MarketTrend_1', 'Label', 'Profit']
        # self.featuresSel = ['TimeStamp', 'Type', 'TypeCode', 'PeriodCode', 'Dir', 'MarketTrendCode']
        self.predFeatures = ['TypeCode']
        self.nfeatures = len(self.featuresSel)

        # specify features by type
        self.CONTINUOUS_COLS = []
        self.CATEGORICAL_COLS = ['Type', 'Period', 'MarketTrend']

        self.periodDict = {'PERIOD_1_MIN': 1, 'PERIOD_5_MIN': 5, 'PERIOD_15_MIN': 15, 'PERIOD_60_MIN': 60,
                           'PERIOD_TODAY': 1440}

        self.periodCodeDict = {'1': 1, '5': 5, '15': 15, '60': 60, 'D': 1440}

        self.alertTrendDict = {'Neutral': 1, 'Bullish': 2, 'Bearish': 3}

        self.marketTrendDict = {
            'Unknown': 1,
            'Neutral': 2,
            'RangeBound': 3,
            'UptrendStarting': 4,
            'Uptrend': 5,
            'UptrendPullback': 6,
            'UptrendEnding': 7,
            'DowntrendStarting': 8,
            'Downtrend': 9,
            'DowntrendPullback': 10,
            'DowntrendEnding': 11,
            'UptrendContinuing': 12,
            'DowntrendContinuing': 13}

        self.eventCodes = [
            'BBLU', 'BBLD', 'BBHU', 'BBHD',

            'HILMFU','HILMFD', 'HILFTU', 'HILFTD',

            'MACDBBHU','MACDBBHD', 'MACDBBLU', 'MACDBBLD','MACDBBTU','MACDBBTD',

            'PRCHU','PRCHD','PRCLU','PRCLD',
            'PRCSHU', 'PRCSHD','PRCSLU', 'PRCSLD',
            'PRCMHU', 'PRCMHD', 'PRCMLU', 'PRCMLD',

            'PRCLHU','PRCLHD', 'PRCLLU','PRCLLD',

            'SARSU','SARSD', 'SARRU', 'SARRD',

            'RSI20U', 'RSI20D', 'RSI80U', 'RSI80D',

            'MFI20U', 'MFI20D', 'MFI80U', 'MFI80D',

            'EOMU', 'EOMD',

            'SC20', 'SC20U',
            'SC20D', 'SC80',
            'SC80D', 'SC80U']

        self.eventCodeDict = {
            'HEARTB':0,
            'VSX':1,
            'VWAPD':2,
            'VWAPU':3,

            'BBLU':11,
            'BBLD':12,
            'BBHU':13,
            'BBHD':14,

            'HILMFU':21,
            'HILMFD':22,
            'HILFTU':23,
            'HILFTD':24,

            'MACDBBHU':31,
            'MACDBBHD':32,
            'MACDBBLU':33,
            'MACDBBLD':34,
            'MACDBBTU':35,
            'MACDBBTD':36,

            'PRCHU':41,
            'PRCHD':42,
            'PRCLU':43,
            'PRCLD':44,

            'PRCSHU':45,
            'PRCSHD':46,
            'PRCSLU':47,
            'PRCSLD':48,

            'PRCMHU':49,
            'PRCMHD':50,
            'PRCMLU':51,
            'PRCMLD':52,

            'PRCLHU':53,
            'PRCLHD':54,
            'PRCLLU':55,
            'PRCLLD':56,

            'SARSU':61,
            'SARSD':62,
            'SARRU':63,
            'SARRD':64,

            'RSI20U':71,
            'RSI20D':72,
            'RSI80U':73,
            'RSI80D':74,

            'MFI20U':81,
            'MFI20D':82,
            'MFI80U':83,
            'MFI80D':84,

            'EOMU':91,
            'EOMD':92,

            'SC20':101,
            'SC20U':102,
            'SC20D':103,
            'SC80':104,
            'SC80D':105,
            'SC80U':106,

            'LRHD':111,
            'LRHU':112,
            'LRLD':113,
            'LRLU':114,

            'GARTD':121,
            'GARTU':122
            }
        self.embedding_size = 32    # tbd
        self.vocab_size = len(self.marketTrendDict.values())
