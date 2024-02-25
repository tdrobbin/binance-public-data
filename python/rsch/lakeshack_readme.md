# LakeShack
_Not Quite a LakeHouse_

LakeShack makes multiple deltalake tables available through SQL by using delta-rs, arrow, and duckdb. Everything runs in-process and doesn't require a spark cluster. Currently only supports storage on local filesystem.

# Installation
Via pip
```bash
pip install lakeshack
```

Via Conda
```bash
conda install lakeshack -c conda-forge
```
# Basic Usage

`LakeShack.write_deltalake` is the main function to write to delta lake tables in the lakeshack. It mirrors `deltalake.write_deltalake` but also registers the table with the in memeory duckdb sql instance attached to the lakeshack so it can be queried from sql.

```python
from lakeshack import LakeShack
import pandas as pd

# connect to existing lakeshak or create if one doesn't exist
lakeshack = LakeShack('/path/to/lakeshack')

data_people = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
})
lakeshack.write_deltalake('people', data_people)

data_purchases = pd.DataFrame({
    'purchase_id': [101, 102, 103, 104, 105],
    'buyer_id': [3, 1, 2, 5, 4],
    'amount': [250, 75, 120, 150, 200]
})
lakeshack.write_deltalake('purchases', data_purchases)

# Perform a join query between 'people' and 'purchases'
join_query = """
SELECT p.name, pu.amount
FROM people p
JOIN purchases pu ON p.id = pu.buyer_id
"""
join_result = lakeshack.sql(join_query)
```

Table names that are not valid SQL can be accessed by using quotes.

```python
lakeshack.write_deltalake('some table name with spaces', data)
lakeshack.sql('select * from "some table name with spaces"')
```


# Managing Delta Tables
All delta tables are available via the `LakeShack.tables` attribute. This is a dictionary of `DeltaTable` instances.

```python
# Get a specific DeltaTable instance
people_table = lakeshack.tables['people']

# Access table properties
table_name = people_table.name
table_path = people_table.path
table_schema = people_table.schema

# Perform operations on the DeltaTable
table_version = people_table.version()
table_files = people_table.files()
table_history = people_table.history()

# delete rows
people_table.delete('amount > 200')
lakeshack.sql('select * from people')

# load previous version before deletes
people_table.load_version(0)
lakeshack.sql('select * from people')

# back to present version after deletes
people_table.load_with_datetime(str(pd.Timestamp.now()))
lakeshack.sql('select * from people')
```

