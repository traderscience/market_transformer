# Market Transformer

This model is trained with sequences of event codes that represent important price movements in a financial time series. It creates an output sequence of event codes for a specific trading instrument.

####Input Sequences 
 Input sequences are created from several datasets that provide a broader view of overall market state.
 Each element of the sequence contains a timestamp with Symbol and Event info, plus numerical and categorical features.
 Each element also contains a timeframe code (1m to D-aily). 
 Higher timeframe records should carry more significance to future movements than lower timeframe records.
 
The EventCode is used to build a vocabulary and embedding table for the model.

#### Input data columns (subject to change)
TimeStamp, Symbol, Exchange, EventCode, Timeframe, RecordType (1-10), EventType, DirectionCode (-1,0,1), [numerical features], [categorical features]

#####Dataset 1
`007/01/02T21:00:00,1WLIB.X,UNDEF,VSX_D,1001,1,VS,0,5.31,0,0,0,0,0,0,0,0,0,0,0,92.86,95.24,95.24,1,7,2,2,2,2
2007/01/02T21:00:00,1WLIB.X,UNDEF,SC20D_D,1001,60,SC20,-1,5.31,0,0,0,0,0,0,0,0,0,0,0,92.86,95.24,95.24,10,7,2,2,2,2
2007/01/02T21:00:00,1WLIB.X,UNDEF,MACDBBHD_D,1001,28,MACDBBH,-1,5.31,0,0,0,0,0,0,0,0,0,0,0,92.86,95.24,95.24,7,7,2,2,2,2
2007/01/03T21:00:00,1WLIB.X,UNDEF,VSX_D,1001,1,VS,0,5.31,0,0,0,0,0,0,0,0,0,0,0,92.86,95.24,95.24,1,7,2,2,2,2
2007/01/04T21:00:00,1WLIB.X,UNDEF,VSX_D,1001,1,VS,0,5.31,0,0,0,0,0,0,0,0,0,0,0,92.86,95.24,95.24,1,2,2,2,2,2
2007/01/05T21:00:00,1WLIB.X,UNDEF,VSX_D,1001,1,VS,0,5.30,0,0,0,0,0,0,0,0,0,0,0,92.86,95.24,95.24,1,2,2,2,2,2
2007/01/05T21:00:00,1WLIB.X,UNDEF,HILXSU_D,1001,12,HILXS,1,5.30,0,0,0,0,0,0,0,0,0,0,0,92.86,95.24,95.24,3,2,2,2,2,2
2007/01/05T21:00:00,1WLIB.X,UNDEF,HILFTD_D,1001,18,HILFT,-1,5.30,0,0,0,0,0,0,0,0,0,0,0,92.86,95.24,95.24,7,2,2,2,2,2
2007/01/08T21:00:00,1WLIB.X,UNDEF,VSX_D,1001,1,VS,0,5.30,0,0,0,0,0,0,0,0,0,0,0,92.86,95.24,95.24,1,2,2,2,2,2
2007/01/09T21:00:00,1WLIB.X,UNDEF,VSX_D,1001,1,VS,0,5.31,0,0,0,0,0,0,0,0,0,0,0,92.86,95.24,95.24,1,8,2,2,2,2

#####Dataset 2
2007/01/03T21:00:00,VIX.XO,CBOE,VSX_D,1001,1,VS,0,12.04,-0.6645,0,0,0,0,0,100.00,100.00,95.00,37.25,34.14,34.14,14.04,31.29,1,4,2,2,2,2
2007/01/03T21:00:00,VIX.XO,CBOE,HILSINX_D,1001,14,HILSIN,0,12.04,-0.6645,0,0,0,0,0,100.00,100.00,100.00,37.25,34.14,34.14,14.04,31.29,1,4,2,2,2,2
2007/01/03T21:00:00,VIX.XO,CBOE,HILFTU_D,1001,17,HILFT,1,12.04,-0.6645,0,0,0,0,0,100.00,100.00,100.00,37.25,34.14,34.14,14.04,31.29,3,4,2,2,2,2
2007/01/04T21:00:00,VIX.XO,CBOE,VWAPD_D,1001,64,VWAP,-1,11.51,-3.41,0,0,0,0,0,100.00,100.00,95.00,37.25,34.14,34.14,14.04,31.29,7,4,2,2,2,2
2007/01/04T21:00:00,VIX.XO,CBOE,VSX_D,1001,1,VS,0,11.51,-3.41,0,0,0,0,0,100.00,100.00,95.00,37.25,34.14,34.14,14.04,31.29,1,4,2,2,2,2
2007/01/04T21:00:00,VIX.XO,CBOE,MACDBBHU_D,1001,27,MACDBBH,1,11.51,-3.41,0,0,0,0,0,100.00,100.00,95.00,37.25,34.14,34.14,14.04,31.29,2,4,2,2,2,2
2007/01/05T21:00:00,VIX.XO,CBOE,VWAPU_D,1001,63,VWAP,1,12.14,1.34,0,0,0,0,0,100.00,100.00,95.00,37.25,34.14,34.14,14.04,31.29,3,4,2,2,2,2
2007/01/05T21:00:00,VIX.XO,CBOE,VSX_D,1001,1,VS,0,12.14,1.34,0,0,0,0,0,100.00,100.00,95.00,37.25,34.14,34.14,14.04,31.29,1,4,2,2,2,2
2007/01/08T21:00:00,VIX.XO,CBOE,VSX_D,1001,1,VS,0,12.00,-2.27,0,0,0,0,0,100.00,100.00,95.00,37.25,34.14,34.14,14.04,31.29,1,4,2,2,2,2
2007/01/09T21:00:00,VIX.XO,CBOE,VWAPD_D,1001,64,VWAP,-1,11.91,-0.6087,0,0,0,0,0,100.00,100.00,95.00,37.25,34.14,34.14,14.04,31.29,7,4,2,2,2,2
`

#####Dataset 3
2007/01/03T04:59:00,DX.X,ICE,VWAPD_D,1001,64,VWAP,-1,83.20,-0.1863,0,0,0,0,0,100.00,100.00,100.00,22.22,51.91,51.91,46.26,33.01,7,7,2,2,2,2
2007/01/03T04:59:00,DX.X,ICE,VSX_D,1001,1,VS,0,83.20,-0.1863,0,0,0,0,0,100.00,100.00,100.00,22.22,51.91,51.91,46.26,33.01,1,7,2,2,2,2
2007/01/03T04:59:00,DX.X,ICE,LRHD_D,1001,20,LRH,-1,83.20,-0.1863,0,0,0,0,0,100.00,100.00,100.00,22.22,51.91,51.91,46.26,33.01,6,7,2,2,2,2
2007/01/03T04:59:00,DX.X,ICE,HILMFD_D,1001,16,HILMF,-1,83.20,-0.1863,0,0,0,0,0,100.00,100.00,100.00,22.22,51.91,51.91,46.26,33.01,8,7,2,2,2,2
2007/01/04T04:59:00,DX.X,ICE,VWAPU_D,1001,63,VWAP,1,83.92,0.3813,0,0,0,0,0,100.00,100.00,100.00,22.22,51.91,51.91,46.26,33.01,3,5,2,2,2,2
2007/01/04T04:59:00,DX.X,ICE,VSX_D,1001,1,VS,0,83.92,0.3813,0,0,0,0,0,100.00,100.00,100.00,22.22,51.91,51.91,46.26,33.01,1,5,2,2,2,2
2007/01/04T04:59:00,DX.X,ICE,MACDBBTU_D,1001,31,MACDBBT,1,83.92,0.3813,0,0,0,0,0,100.00,100.00,100.00,22.22,51.91,51.91,46.26,33.01,3,5,2,2,2,2
2007/01/04T04:59:00,DX.X,ICE,LRHU_D,1001,19,LRH,1,83.92,0.3813,0,0,0,0,0,100.00,100.00,100.00,22.22,51.91,51.91,46.26,33.01,6,5,2,2,2,2
2007/01/04T04:59:00,DX.X,ICE,HILPFU_D,1001,10,HILPF,1,83.92,0.3813,0,0,0,0,0,100.00,100.00,100.00,22.22,51.91,51.91,46.26,33.01,3,5,2,2,2,2
2007/01/05T04:59:00,DX.X,ICE,VWAPU_D,1001,63,VWAP,1,84.32,0.2431,0,0,0,0,0,100.00,100.00,100.00,22.22,51.91,51.91,46.26,33.01,3,4,2,2,2,2

### Preprocessing
Market data is continuous, but sequence lengths for training and inference must be limited to be manageable. 
The  approach taken is to select the most recent 3 records for each timeframe from each dataset, and merge them into
 one output series in time order.
Categorical features need to be one-hot encoded.
Numerical features need to be standardized based on the historical range of each column over a 12 year period for each symbol.

### Sample Output Sequence
The output sequence will contain the predicted series of EventCodes.
For example:
HILMFU_5
MACDBBLU_5
HILFTU_5
MACDBBHU_5
SC80U_5
BBHU_15

## Training
The input training data sequences will contain content from multiple datasets, eg. VIX, USD, Oil.
The label sequences provided for training will represent a single symbol only, eg. "AAPL".

