# Reduce Memory Usage 
    
To optimize the memory usage of a Pandas DataFrame object, it is important to convert different data types to their proper format. For instance, a column such as ro_code in a DataFrame may have a default data type of int64, but the int32 data type is often sufficient to store data without altering the values. By converting this column to int32, we can free up RAM space and improve the memory efficiency of the DataFrame.

## Import library


```python
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level= logging.INFO,logger='app')
logger = logging.getLogger('app')
```


```python
df = pd.read_csv('AAPL10Y.csv')
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2517 entries, 0 to 2516
    Data columns (total 6 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   date    2517 non-null   object 
     1   close   2517 non-null   float64
     2   volume  2517 non-null   float64
     3   open    2517 non-null   float64
     4   high    2517 non-null   float64
     5   low     2517 non-null   float64
    dtypes: float64(5), object(1)
    memory usage: 118.1+ KB


## Reduce Memory Usage function


```python
def reduce_memory_usage(df, printf=True):
    """
    reduce_memory_usage 
    Parameters
    ----------
    df : dataframe

    Returns
    -------
    df : dataframe

    """

    intial_memory = round(df.memory_usage().sum() / 1024**2,2)
    printf_info = f"Intial memory usage: {intial_memory} MB"
    if printf:
        logger.info(printf_info)
    else:
        print(printf_info)

    for col in df.columns:
        if df[col].dtype != object:
            mn = df[col].min()
            mx = df[col].max()
            if df[col].dtype == int:
                if mn >= 0:
                    if mx < np.iinfo(np.uint8).max:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < np.iinfo(np.uint16).max:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < np.iinfo(np.uint32).max:
                        df[col] = df[col].astype(np.uint32)
                    elif mx < np.iinfo(np.uint64).max:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
            if df[col].dtype == float:
                # if mn > np.finfo(np.float16).min and mx < np.finfo(np.float16).max:
                #     df[col] = df[col].astype(np.float16)
                if mn > np.finfo(np.float32).min and mx < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    # log memory reduction
    red_memory = df.memory_usage().sum() / 1024**2
    percent = (intial_memory - red_memory) * 100 / intial_memory
    printf_info = f"Memory usage decreased to {red_memory:.2f} Mb ({percent:.2f}% reduction)"
    if printf:
        logger.info(printf_info)
    else:
        print(printf_info)

    return df
```


```python
df = reduce_memory_usage(df, printf=True)
```

    INFO:app:Intial memory usage: 0.07 MB
    INFO:app:Memory usage decreased to 0.07 Mb (3.81% reduction)



```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2517 entries, 0 to 2516
    Data columns (total 6 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   date    2517 non-null   object 
     1   close   2517 non-null   float32
     2   volume  2517 non-null   float32
     3   open    2517 non-null   float32
     4   high    2517 non-null   float32
     5   low     2517 non-null   float32
    dtypes: float32(5), object(1)
    memory usage: 68.9+ KB

