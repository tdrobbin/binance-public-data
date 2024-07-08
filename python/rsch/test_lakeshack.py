import pandas as pd
import pytest

import sys; print(sys.path)
import os; print(os.getcwd())

# from lakeshack import LakeShack  # Adjust the import based on your project structure
from tinydb import TinyDB, Query
# from rsch.deltalake import DeltaTable, write_deltalake
from deltalake import DeltaTable, write_deltalake
import duckdb
import tempfile


import pandas as pd
import pytest
import tempfile
from deltalake import DeltaTable
from rsch.lakeshack import LakeShack  # Adjust the import based on your project structure
from tinydb import Query

@pytest.fixture
def lakehouse():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield LakeShack(tmpdirname)

def test_init(lakehouse):
    """Test the initialization of LakeShack."""
    assert lakehouse.path.exists()

def test_tables_property(lakehouse):
    """Test the tables property."""
    assert isinstance(lakehouse.tables, dict)

def test_write_and_read(lakehouse):
    """Test writing to and reading from a Delta Lake table."""
    df_write = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    table_name = 'test_table'
    lakehouse.write_deltalake(table_name, df_write)
    # lakehouse._update_sql_registry()
    df_read = lakehouse.sql(f"SELECT * FROM {table_name}")
    pd.testing.assert_frame_equal(df_read, df_write)

def test_getitem(lakehouse):
    """Test the __getitem__ method."""
    table_name = 'test_table'
    df_write = pd.DataFrame({'a': [1, 2, 3]})
    lakehouse.write_deltalake(table_name, df_write)
    # lakehouse._update_sql_registry()
    assert isinstance(lakehouse[table_name], DeltaTable)

def test_table_creation_and_registration(lakehouse):
    """Test table creation and registration in DuckDB."""
    df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
    table_name = 'new_table'
    lakehouse.write_deltalake(table_name, df)
    assert table_name in lakehouse.tables
    registered_tables = lakehouse.sql('show tables').name.values
    assert table_name in registered_tables

def test_data_integrity(lakehouse):
    """Test the integrity of data written and read."""
    df_write = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
    table_name = 'integrity_test_table'
    lakehouse.write_deltalake(table_name, df_write)
    df_read = lakehouse.sql(f"SELECT * FROM {table_name}")
    pd.testing.assert_frame_equal(df_read.reset_index(drop=True), df_write)

def test_sql_query(lakehouse):
    """Test executing SQL queries."""
    table_name = 'query_test_table'
    df = pd.DataFrame({'num': [10, 20, 30]})
    lakehouse.write_deltalake(table_name, df)
    result = lakehouse.sql(f"SELECT num FROM {table_name} WHERE num > 15")
    expected = pd.DataFrame({'num': [20, 30]})
    pd.testing.assert_frame_equal(result, expected)

@pytest.mark.skip("not currently implemented")
def test_metadata_db_update(lakehouse):
    """Test updating the metadata database."""
    table_name = 'metadata_test_table'
    df = pd.DataFrame({'data': [100, 200, 300]})
    lakehouse.write_deltalake(table_name, df)
    metadata = lakehouse.metadata_db.search(Query().name == table_name)
    assert len(metadata) == 1
    assert metadata[0]['path'] == str(lakehouse._get_table_path(table_name))

def test_empty_dataframe_write_and_read(lakehouse):
    """Test writing and reading an empty DataFrame with one column."""
    df_empty = pd.DataFrame({'dummy_column': pd.Series(dtype='int')})
    table_name = 'empty_table'
    lakehouse.write_deltalake(table_name, df_empty)
    # lakehouse._update_sql_registry()
    df_read = lakehouse.sql(f"SELECT * FROM {table_name}")
    pd.testing.assert_frame_equal(df_read, df_empty)


def test_sql_query_no_results(lakehouse):
    """Test SQL query that should return no results."""
    df = pd.DataFrame({'number': [1, 2, 3]})
    table_name = 'no_results_table'
    lakehouse.write_deltalake(table_name, df)
    result = lakehouse.sql(f"SELECT * FROM {table_name} WHERE number > 100")
    assert result.empty

def test_access_non_existent_table(lakehouse):
    """Test that accessing a non-existent table raises an exception."""
    non_existent_table_name = "non_existent_table"

    # Attempt to access a non-existent table
    with pytest.raises(KeyError):
        _ = lakehouse[non_existent_table_name]

def test_update_rows(lakehouse):
    """Test updating a table."""
    table_name = 'update_table'
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    lakehouse.write_deltalake(table_name, df)

    lakehouse[table_name].update({'x': '10'}, predicate= 'y = 4')
    df_read = lakehouse.sql(f"SELECT * FROM {table_name}")
    df_update = pd.DataFrame({'x': [10, 2, 3], 'y': [4, 5, 6]})
    pd.testing.assert_frame_equal(df_read.reset_index(drop=True), df_update)

def test_delete_rows(lakehouse):
    """Test deleting rows from a table."""
    table_name = 'delete_table'
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    lakehouse.write_deltalake(table_name, df)

    lakehouse[table_name].delete(predicate= 'y = 4')
    df_read = lakehouse.sql(f"SELECT * FROM {table_name}")
    df_update = pd.DataFrame({'x': [2, 3], 'y': [5, 6]})
    pd.testing.assert_frame_equal(df_read.reset_index(drop=True), df_update)


    # df_update = pd.DataFrame({'x': [10, 20, 30], 'y': [40, 50, 60]})
    # lakehouse.write_deltalake(table_name, df_update)
    # df_read = lakehouse.sql(f"SELECT * FROM {table_name}")
    # pd.testing.assert_frame_equal(df_read.reset_index(drop=True), df_update)