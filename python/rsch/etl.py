from utility import get_all_symbols
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import os
# from utility import get_path
import utility
from tqdm.auto import tqdm
# from arcticdb import Arctic
from deltalake import DeltaTable, write_deltalake
import duckdb
import zipfile
import pyarrow.csv as pc
import pyarrow as pa
from typing import List, Union, Tuple, Optional, Dict, Any, Callable, Iterable
import re

# from .lakeshack import LakeShack


def get_download_command():
    # resp = requests.get('https://fapi.binance.com/fapi/v1/exchangeInfo')
    syms = get_all_symbols('um')
    tgt_syms = [s for s in syms if s.endswith('USDT')]
    cmd = """python3 download-kline.py -t um -i 1m -skip-daily 1 -startDate 2013-01-01 -endDate 2023-03-31 -skip-daily 1 -s """
    cmd = cmd + ' '.join(tgt_syms)

    return cmd


def get_store_directory():
    store_directory = os.environ.get('STORE_DIRECTORY')
    if not store_directory:
        # store_directory = os.path.dirname(os.path.realpath(utility.__file__))
        # use Path to get absolute path
        store_directory = Path(utility.__file__).parent.absolute()

    return Path(store_directory)


def get_arcticdb_uri():
    ARCTICDB_FPATH = get_store_directory() / 'data' / Path('arcticdb')
    ARCTICDB_URI = f'lmdb://{ARCTICDB_FPATH}'

    return ARCTICDB_URI


def get_deltalake_uri():
    DELTALAKE_FPATH = get_store_directory() / 'data' / Path('deltalake')
    DELTALAKE_FPATH = get_store_directory() / 'data' / Path('deltalake_v2')
    DELTALAKE_FPATH = get_store_directory() / 'data' / Path('deltalake_v3')

    DELTALAKE_URI = DELTALAKE_FPATH
    # DELTALAKE_URI = f'file://{DELTALAKE_FPATH}'

    return DELTALAKE_URI


def get_deltalake_table_path(
    trading_type='um',
    market_data_type='klines',
    time_period: Optional[str]='monthly',
    symbol: Optional[str]='BTCUSDT',
    interval: Optional[str]='1m',
):
    if market_data_type == 'klines':
        # klines are partitioned by symbol
        return get_deltalake_uri() / f'{trading_type}_{market_data_type}_{interval}'
    
    elif market_data_type == 'trades':
        return get_deltalake_uri() / f'{trading_type}_{market_data_type}'
    
    else:
        raise NotImplementedError


def get_all_symbols_in_data_direcory(
    trading_type='um',
    market_data_type='klines',
    time_period: Optional[str]='monthly'
) -> List[str]:
    """
    get all symbols in the raw data directory. this is the directory where the data is downloaded to
    before it is loaded into arcticdb or deltalake.
    """
    trading_type_path = 'data/spot'
    if trading_type != 'spot':
        trading_type_path = f'data/futures/{trading_type}'
    
    symbol_dr = get_store_directory() / Path(f'{trading_type_path}/{time_period}/{market_data_type}')
    symbols = [p.name for p in symbol_dr.glob('*') if p.is_dir()]

    return symbols


def get_all_filenames_in_data_directory(
    trading_type='um',
    market_data_type='klines',
    time_period: Optional[str]='monthly',
    symbol: Optional[str]='BTCUSDT',
    interval: Optional[str]='1m',
):
    example_filepath = utility.get_path(
        trading_type=trading_type,
        market_data_type=market_data_type,
        time_period=time_period,
        symbol=symbol,
        interval=interval if market_data_type == 'klines' else None
    )

    file_dir = get_store_directory() / Path(example_filepath)
    # print(file_dir)
    # return list(file_dir.glob('*.zip')
    # filenames = [p.name for p in file_dir.glob('*.zip') if p.is_file()]
    filenames = [p for p in file_dir.glob('*.zip') if p.is_file()]

    return filenames


def get_data_schemas() -> dict:
    schemas = """
    spot:
        aggtrades: '|Aggregate tradeId|Price|Quantity|First tradeId|Last tradeId|Timestamp|Was the buyer the maker|Was the trade the best price match|'
        trades: '|trade Id| price| qty|quoteQty|time|isBuyerMaker|isBestMatch|'
        klines: '|Open time|Open|High|Low|Close|Volume|Close time|Quote asset volume|Number of trades|Taker buy base asset volume|Taker buy quote asset volume|Ignore|'
        trades: '|trade Id|price|qty|quoteQty|time|isBuyerMaker|isBestMatch|'
    um:
        aggtrades: '|Aggregate tradeId|Price|Quantity|First tradeId|Last tradeId|Timestamp|Was the buyer the maker|'
        trades: '|trade Id| price| qty|quoteQty|time|isBuyerMaker'
        klines: '|Open time|Open|High|Low|Close|Volume|Close time|Quote asset volume|Number of trades|Taker buy base asset volume|Taker buy quote asset volume|Ignore|'
        trades: '|trade Id|price|qty|quoteQty|time|isBuyerMaker|'
    cm:
        aggtrades: '|Aggregate tradeId|Price|Quantity|First tradeId|Last tradeId|Timestamp|Was the buyer the maker|'
        trades: '|trade Id| price| qty|quoteQty|time|isBuyerMaker|'
        klines: '|Open time|Open|High|Low|Close|Volume|Close time|Base asset volume|Number of trades|Taker buy volume|Taker buy base asset volume|Ignore|'
        trades: '|trade Id|price|qty|baseQty|time|isBuyerMaker|'
    """
    schemas = yaml.safe_load(schemas)

    def _to_snake_case(name):
        # Replace spaces and camelCase with snake_case
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
        return s2.replace(' ', '_').lower()

    def _modify_column_names(schemas):
        for key in schemas:
            for sub_key in schemas[key]:
                # Split columns, apply snake_case conversion, and rejoin
                columns = schemas[key][sub_key].strip('|').split('|')
                new_columns = [_to_snake_case(col.strip()) for col in columns]
                schemas[key][sub_key] = '|' + '|'.join(new_columns) + '|'

    _modify_column_names(schemas)

    return schemas


def get_secid_data(
    trading_type: str = 'um', 
    market_data_type: str = 'klines', 
    time_period: str = 'monthly', 
    symbol: str = 'BTCUSDT', 
    interval: str = '1m',
    filename_wildcards: str | Iterable[str] = None
) -> pd.DataFrame:
    data_path = utility.get_path(
        trading_type=trading_type,
        market_data_type=market_data_type,
        time_period=time_period,
        symbol=symbol,
        interval=interval
    )
    """

    """

    schemas = get_data_schemas()

    store_directory = get_store_directory()
    data_path = store_directory / Path(data_path)
    table_name = f"{trading_type}.{market_data_type}.{time_period}.{interval}"

    def _get_wildcard(filename):
        return f"*{filename}" if filename.endswith('.zip') else f"*{filename}.zip"

    if filename_wildcards is not None:
        if isinstance(filename_wildcards, str):
            filename_wildcards = [filename_wildcards]

        if isinstance(filename_wildcards, Iterable) and all(isinstance(item, str) for item in filename_wildcards):
            tgt_fpaths = []
            for period in filename_wildcards:
                fpath_wildcard = _get_wildcard(period)
                tgt_fpaths.extend(data_path.glob(fpath_wildcard))
        else:
            raise ValueError('time_period_filename must be a string or an iterable of strings')
    else:
        tgt_fpaths = list(data_path.glob('*.zip'))

    dfs = []
    # for fpath in data_path.glob(fpath_wildard):
    for fpath in tqdm(tgt_fpaths, disable=True):
        # # dfi = pd.read_csv(fpath, engine='pyarrow', header=0)
        # dfi = pd.read_csv(fpath, header=0)
        # # if value in first column of first row is a string drop the first row. but may be
        # # a number that was parsed as a string so check if it can be converted to a float
        # if pd.notnull(pd.to_numeric(dfi.columns[0], errors='coerce')):
        #     dfi = pd.read_csv(fpath, header=None)
        
        dfi = pd.read_csv(fpath, header=0)

        # Check if the value in the first column of the first row is a string that can't be converted to a float
        try:
            float(dfi.iloc[0, 0])
            is_data = True
        except ValueError:
            is_data = False
        except IndexError:
            # no data/bad data so skip
            continue

        # If the first row is a header, read the CSV again without headers
        if not is_data:
            dfi = pd.read_csv(fpath, header=None)

        dfi.columns = [c.lower().replace(' ', '_') for c in schemas[trading_type][market_data_type].split('|') if c != '']
        # dfi = dfi[[c for c in dfi.columns if c != 'ignore' and c != '' and c is not None and isinstance(c, str)]]
        dfi = dfi.rename(columns=lambda x: str(x).strip().lower().replace(' ', '_'))
        # get yyyy-mm in the filename and convert to an int in the form of yyyymm and add as 'filename' column
        # e.g. BTCUSDT-trades-2020-02.zip -> 202002
        dfi['month'] = int(re.findall(r'\d{4}-\d{2}', fpath.name)[0].replace('-', ''))
        dfi['filepath'] = str(fpath)
        dfi['filename'] = fpath.name
        dfi['tablename'] = table_name
        
        dfs.append(dfi)

    if not dfs:
        raise FileNotFoundError(f'no files found in: {data_path}')
    
    df = pd.concat(dfs)
    df.insert(0, 'secid', symbol)

    for col in df.columns:
        if 'time' in col:
            df[col] = pd.to_datetime(df[col], unit='ms', origin='unix')
            df = df.dropna(subset=[col])
    
    for col in df.columns:
        if 'time' in col:
            continue

        elif col == 'ignore':
            df = df.drop(columns=col)

        elif col.lower() in ['isbuyermaker', 'isbestmatch', 'isbuyermaker']:
            df[col] = df[col].astype(bool)

        elif col in ['trade_id', 'month', 'number_of_trades']:
            df[col] = df[col].astype(int)

        elif col in ['filename', 'filepath', 'tablename', 'secid', 'symbol']:
            df[col] = df[col].astype('string')

        else:
            df[col] = df[col].astype(np.float64)
    
    some_timestamp_col = [c for c in df.columns if 'time' in c][0]
    df = df.sort_values(by=some_timestamp_col, ascending=True)
    
    return df


def update_lakeshack(
    path: str | Path,
    trading_type='um',
    market_data_type='klines',
    time_period='monthly',
    symbols=None,
    interval='1m'
):
    # ls = LakeShack(path)

    table_name = f"{trading_type}_{market_data_type}_{time_period}_{interval}"
    delta_table_path = Path(path) / table_name

    if market_data_type == 'klines':
        if interval == '1m':
            partition_by = ['secid']
        else:
            partition_by = None
    elif market_data_type in ['trades', 'aggTrades']:
        partition_by = ['secid', 'month']

    if symbols is None:
        symbols = get_all_symbols_in_data_direcory(
            trading_type=trading_type,
            market_data_type=market_data_type,
            time_period=time_period
        )
        import random
        random.shuffle(symbols)

    for s in tqdm(symbols):
        # if table_name not in ls.tables:
        if not delta_table_path.exists():
            df = get_secid_data(
                trading_type=trading_type,
                market_data_type=market_data_type,
                time_period=time_period,
                symbol=s,
                interval=interval,
                filename_wildcards=None
            )
        else:
            fpaths = get_all_filenames_in_data_directory(
                trading_type=trading_type,
                market_data_type=market_data_type,
                time_period=time_period,
                symbol=s,
                interval=interval
            )
            fetched_fnames = [f.name for f in fpaths]


            # loaded_fnames = ls.sql(f"select distinct filename from '{table_name}' where secid = '{s}'").filename.to_list()
            con = duckdb.connect()
            loaded_fnames = con.sql(f"""select distinct filename from delta_scan('{delta_table_path}') where secid = '{s}'""").df().filename.to_list()

            new_fnames = [f for f in fetched_fnames if f not in loaded_fnames]

            if len(new_fnames) == 0:
                continue
            
            df = get_secid_data(
                trading_type=trading_type,
                market_data_type=market_data_type,
                time_period=time_period,
                symbol=s,
                interval=interval,
                filename_wildcards=new_fnames
            )

        # print(f"partition_by: {partition_by}")
        # ls.write_deltalake(
        #     table_name=table_name,
        #     data=df,
        #     mode='append',
        #     partition_by=partition_by
        # )


        print(s)
        data = pa.Table.from_pandas(df, preserve_index=False)
        write_deltalake(
            delta_table_path,
            data=data,
            mode='append',
            # partition_by=partition_by
        )

    # ls.tables[table_name].optimize.compact()
    # ls._update_tables()

    print(f"Updated {table_name}")


def load_secid_data_to_deltalake__deprecated(
        trading_type='um',
        market_data_type='klines',
        time_period='monthly',
        symbol='BTCUSDT',
        interval='1m',
        time_period_filename=None
):
    """
    goes through the entire data directories in `futures` and `spot` and loads
    all data by secid into a deltalake table. for market_data_type `klines` there is one table per
    trading_type, market_data_type, and interval. these tables are partitioned by
    symbol. market_data_type `trades` there is one table per symbol.

    the base dir is from get_store_directory(). the dta in there is stored according to the
    `utility.get_path` function:

    def get_path(trading_type, market_data_type, time_period, symbol, interval=None):
        trading_type_path = 'data/spot'
        if trading_type != 'spot':
            trading_type_path = f'data/futures/{trading_type}'
        if interval is not None:
            path = f'{trading_type_path}/{time_period}/{market_data_type}/{symbol.upper()}/{interval}/'
        else:
            path = f'{trading_type_path}/{time_period}/{market_data_type}/{symbol.upper()}/'
        return path
    """
    if market_data_type == 'klines':
        partition_cols = ['secid']
    elif market_data_type == 'trades':
        partition_cols = ['secid', 'month']
    else:
        raise NotImplementedError()

    table_path = get_deltalake_table_path(
        trading_type=trading_type, 
        market_data_type=market_data_type, 
        interval=interval
    )

    try:
        df = get_secid_data(
            trading_type=trading_type,
            market_data_type=market_data_type,
            time_period=time_period,
            symbol=symbol,
            interval=interval,
            filename_wildcards=time_period_filename
        )
    except FileNotFoundError as e:
        print(e)
        return

    pa_table = pa.Table.from_pandas(df, preserve_index=False)

    write_deltalake(table_path, pa_table, partition_by=partition_cols, mode='append')

    


def make_vwap_duckdb__deprecated(trades) -> pd.DataFrame:
    trades = trades
    qry = """
    WITH PriceQty AS (
        SELECT 
            time,
            price * qty AS price_x_qty,
            qty
        FROM trades
    ),
    Resampled AS (
        SELECT 
            date_trunc('minute', time) AS time_minute,
            SUM(price_x_qty) AS sum_price_x_qty,
            SUM(qty) AS sum_qty
        FROM PriceQty
        GROUP BY time_minute
    )
    SELECT 
        time_minute,
        sum_price_x_qty,
        sum_qty,
        sum_price_x_qty / NULLIF(sum_qty, 0) AS vwap
    FROM Resampled;
    """
    vwap_duckdb = duckdb.sql(qry).df()

    return vwap_duckdb




# -- Trash --

# def load_klines_to_arcticdb_eg():
#     ac = Arctic(ARCTICDB_URI)
    
#     lib_name = 'futures_um.klines.1m.secid'
#     # ac.create_library(lib_name)

#     tgt_syms = get_all_symbols('um')
#     for sym in tqdm(tgt_syms):
#         df = load_secid_data(
#             trading_type='um', 
#             market_data_type='klines', 
#             time_period='monthly', 
#             symbol=sym, 
#             interval='1m'
#         )
#         df = df.set_index('open_time', drop=False)
#         meta = {'write_timestamp': pd.Timestamp.utcnow()}

#         ac[lib_name].write(sym, df, metadata=meta)


# def load_secid_data__old(
#     trading_type='um', 
#     market_data_type='klines', 
#     time_period='monthly', 
#     symbol='BTCUSDT', 
#     interval='1m',
#     time_period_filename=None
# ) -> pd.DataFrame:
#     data_path = utility.get_path(
#         trading_type=trading_type,
#         market_data_type=market_data_type,
#         time_period=time_period,
#         symbol=symbol,
#         interval=interval
#     )

#     store_directory = get_store_directory()
#     data_path = store_directory / Path(data_path)

    # schemas = """
    # spot:
    #     aggtrades: '|Aggregate tradeId|Price|Quantity|First tradeId|Last tradeId|Timestamp|Was the buyer the maker|Was the trade the best price match|'
    #     trades: '|trade Id| price| qty|quoteQty|time|isBuyerMaker|isBestMatch|'
    #     klines: '|Open time|Open|High|Low|Close|Volume|Close time|Quote asset volume|Number of trades|Taker buy base asset volume|Taker buy quote asset volume|Ignore|'
    #     trades: '|trade Id|price|qty|quoteQty|time|isBuyerMaker|isBestMatch|'
    # um:
    #     aggtrades: '|Aggregate tradeId|Price|Quantity|First tradeId|Last tradeId|Timestamp|Was the buyer the maker|'
    #     trades: '|trade Id| price| qty|quoteQty|time|isBuyerMaker'
    #     klines: '|Open time|Open|High|Low|Close|Volume|Close time|Quote asset volume|Number of trades|Taker buy base asset volume|Taker buy quote asset volume|Ignore|'
    #     trades: '|trade Id|price|qty|quoteQty|time|isBuyerMaker|'
    # cm:
    #     aggtrades: '|Aggregate tradeId|Price|Quantity|First tradeId|Last tradeId|Timestamp|Was the buyer the maker|'
    #     trades: '|trade Id| price| qty|quoteQty|time|isBuyerMaker|'
    #     klines: '|Open time|Open|High|Low|Close|Volume|Close time|Base asset volume|Number of trades|Taker buy volume|Taker buy base asset volume|Ignore|'
    #     trades: '|trade Id|price|qty|baseQty|time|isBuyerMaker|'
    # """
    # schemas = yaml.safe_load(schemas)

#     fpath_wildard = f"*{'' if time_period_filename is None else time_period_filename}.zip"
#     print(fpath_wildard)
#     # df = pd.concat([pd.read_csv(fpath, header=None) for fpath in data_path.glob('*.zip')])
#     df.columns = [c.lower().replace(' ', '_') for c in schemas[trading_type][market_data_type].split('|') if c != '']

#     for col in df.columns:
#         if 'time' in col:
#             df[col] = pd.to_datetime(df[col], unit='ms', origin='unix', errors='coerce')
#             df = df.dropna(subset=[col])
    
#     for col in df.columns:
#         print(col)
#         if 'time' in col:
#             continue

#         elif col == 'ignore':
#             df = df.drop(columns=col)
#             continue

#         elif col.lower() in ['isbuyermaker', 'isbestmatch', 'isbuyermaker']:
#             df[col] = df[col].astype(bool)
#             continue

#         elif col == 'trade_id':
#             df[col] = df[col].astype(int)
#             continue

#         else:
#             df[col] = df[col].astype(np.float64)
    
#     some_timestamp_col = [c for c in df.columns if 'time' in c][0]
#     df = df.sort_values(by=some_timestamp_col, ascending=True)
    
#     return df
    # tables = []
    # for zip_path in data_path.glob(fpath_wildard):
    #     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    #         for csv_filename in zip_ref.namelist():
    #             with zip_ref.open(csv_filename) as csv_file:
    #                 table = pc.read_csv(csv_file)
    #                 # Perform your transformations using PyArrow
    #                 # Example: filtered_table = table.filter(...)
    #                 # tables.append(table)
    #                 tables.append(table.to_pandas())


    # Combine all tables
    # combined_table = pa.concat_tables(tables)
    # df = combined_table.to_pandas()

    # df = pd.concat([table.to_pandas() for table in tables], ignore_index=True)
    # df = pd.concat(tables, ignore_index=True)

# def get_secid_data_pa(
#     trading_type='um', 
#     market_data_type='klines', 
#     time_period='monthly', 
#     symbol='BTCUSDT', 
#     interval='1m',
#     time_period_filename=None
# ) -> pa.Table:
#     data_path = utility.get_path(
#         trading_type=trading_type,
#         market_data_type=market_data_type,
#         time_period=time_period,
#         symbol=symbol,
#         interval=interval
#     )

#     store_directory = get_store_directory()
#     data_path = store_directory / Path(data_path)

#     schemas = """
#     # ... [Your schema YAML content] ...
#     """
#     schemas = yaml.safe_load(schemas)

#     fpath_wildcard = f"*{'' if time_period_filename is None else time_period_filename}.zip"
#     print(fpath_wildcard)

#     tables = []
#     for zip_path in data_path.glob(fpath_wildcard):
#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             for csv_filename in zip_ref.namelist():
#                 with zip_ref.open(csv_filename) as csv_file:
#                     table = pc.read_csv(csv_file)
#                     tables.append(table)

#     # Combine all tables
#     combined_table = pa.concat_tables(tables)

#     # Manipulating columns in PyArrow is different from Pandas. For example:
#     # - To rename columns, create a new table with the renamed columns
#     # - To change data types, cast the columns to new types
#     # - To sort the table, use the `sort_indices` function and then `take` function

#     # Example: Rename columns (assuming your schema provides the new column names)
#     column_names = [c.lower().replace(' ', '_') for c in schemas[trading_type][market_data_type].split('|') if c != '']
#     combined_table = combined_table.rename_columns(column_names)

#     # Example: Convert timestamp columns and sort by a timestamp column
#     timestamp_cols = [col for col in combined_table.column_names if 'time' in col]
#     for col in timestamp_cols:
#         combined_table = combined_table.set_column(combined_table.schema.get_field_index(col), 
#                                                    col, 
#                                                    combined_table[col].cast(pa.timestamp('ms')))
    
#     # Example: Sort by a timestamp column
#     sorted_indices = combined_table[timestamp_cols[0]].sort_indices()
#     combined_table = combined_table.take(sorted_indices)

#     return combined_table

# #def load_secid_data_to_arcticdb():
#     """
#     goes through the entire data directories in `futures` and `spot` and loads
#     all data by secid into arcticdb. creates a new library for each type of data downloaded.
#     the base dir is from get_store_directory(). the dta in there is stored according to the
#     `utility.get_path` function:

#     def get_path(trading_type, market_data_type, time_period, symbol, interval=None):
#         trading_type_path = 'data/spot'
#         if trading_type != 'spot':
#             trading_type_path = f'data/futures/{trading_type}'
#         if interval is not None:
#             path = f'{trading_type_path}/{time_period}/{market_data_type}/{symbol.upper()}/{interval}/'
#         else:
#             path = f'{trading_type_path}/{time_period}/{market_data_type}/{symbol.upper()}/'
#         return path
#     """
#     raise NotImplementedError

    # if time_period_filename is not None:
    #     if isinstance(time_period_filename, str):
    #         fpath_wildcard = f"*{time_period_filename}.zip"
    #         tgt_fpaths = list(data_path.glob(fpath_wildcard))
    #     elif isinstance(time_period_filename, Iterable) and all(isinstance(item, str) for item in time_period_filename):
    #         tgt_fpaths = []
    #         for period in time_period_filename:
    #             fpath_wildcard = f"*{period}.zip"
    #             tgt_fpaths.extend(data_path.glob(fpath_wildcard))
    #     else:
    #         raise ValueError('time_period_filename must be a string or an iterable of strings')
    # else:
    #     tgt_fpaths = list(data_path.glob('*.zip'))

    #     schemas = """
    # spot:
    #     aggtrades: '|Aggregate tradeId|Price|Quantity|First tradeId|Last tradeId|Timestamp|Was the buyer the maker|Was the trade the best price match|'
    #     klines: '|Open time|Open|High|Low|Close|Volume|Close time|Quote asset volume|Number of trades|Taker buy base asset volume|Taker buy quote asset volume|Ignore|'
    #     trades: '|trade Id|price|qty|quoteQty|time|isBuyerMaker|isBestMatch|'
    # um:
    #     aggtrades: '|Aggregate tradeId|Price|Quantity|First tradeId|Last tradeId|Timestamp|Was the buyer the maker|'
    #     klines: '|Open time|Open|High|Low|Close|Volume|Close time|Quote asset volume|Number of trades|Taker buy base asset volume|Taker buy quote asset volume|Ignore|'
    #     trades: '|trade Id|price|qty|quoteQty|time|isBuyerMaker|'
    # cm:
    #     aggtrades: '|Aggregate tradeId|Price|Quantity|First tradeId|Last tradeId|Timestamp|Was the buyer the maker|'
    #     klines: '|Open time|Open|High|Low|Close|Volume|Close time|Base asset volume|Number of trades|Taker buy volume|Taker buy base asset volume|Ignore|'
    #     trades: '|trade Id|price|qty|baseQty|time|isBuyerMaker|'
    # """
    # schemas = yaml.safe_load(schemas)