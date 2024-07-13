__version__ = '0.0.1'


from pathlib import Path
from deltalake import DeltaTable, write_deltalake
from tinydb import TinyDB, Query
import datetime
import deltalake
import duckdb
import pandas as pd
import numpy as np
import pyarrow as pa
import ibis
from typing import Dict, Any, List, Optional, Union, Iterable, Tuple, Callable


# DeltaShack
# DeltaLodge
# LakeLodge
# LakeCabin
class LakeShack:

    def __init__(self, path):
        """
        Initialize the LakeShack instance.

        Args:
            path: The file system path for storing Delta Lake tables.
        """
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.name = self.path.name
        self.metadata_path = self.path / '_metadata.json'
        self.metadata_db = TinyDB(str(self.metadata_path))

        if not self.metadata_db.all():
            self.metadata_db.insert({'metadata': {
                'lakeshack_name': self.name,
                'lakeshack_path': str(self.path),
                'created': datetime.datetime.now().isoformat(),
                'metadata_path': str(self.metadata_path),
                'lakeshack_version': self.__version__,
                'deltalake_version': deltalake.__version__,
            }})
            self.metadata_db.insert({'tables': []})

        self.con = duckdb.connect(database=':memory:', read_only=False)
        self.tables = dict()
        # self.tables = {table_name: DeltaTable(table_data['table_path']) for table_name, table_data in self._on_disk_table_info.items()}
        self._update_tables()

    def __repr__(self):
        return f'LakeShack(name={self.name}, len(tables)={len(self._on_disk_table_info)}, total_size={self.total_size}, path={self.path})'
    
    def __getitem__(self, table_name: str) -> DeltaTable:
        """
        Get the DeltaTable object for a given table name using dictionary-like access.

        Args:
            table_name: The name of the table.

        Returns:
            The DeltaTable object corresponding to the table name.

        Raises:
            KeyError: If the table name does not exist.
        """
        return self.tables[table_name]

    @property
    def _on_disk_table_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a dictionary of Delta Lake tables with their metadata.

        Returns:
            A dictionary where each key is a table name and the value is a dictionary of table metadata.
        """
        table_paths = [p for p in self.path.iterdir() if p.is_dir() and (p / '_delta_log').is_dir()]
        return {
            table_path.name: {
                'table_name': table_path.name,
                'table_path': table_path,
                'table_path_schema': '.'.join(table_path.name.split('.')[:-1]),
            }
            for table_path in table_paths
        }
    
    def _update_tables(self):
        """
        Update the tables property and sql registry with the latest DeltaTable objects.
        """
        for table_name, table_data in self._on_disk_table_info.items():
            # if table_name not in self.tables:
            #     self.tables[table_name] = DeltaTable(table_data['table_path'])
            
            self.tables[table_name] = DeltaTable(table_data['table_path'])

            
            self.con.register(table_name, self.tables[table_name].to_pyarrow_dataset())
    
    @property
    def total_size(self) -> int:
        """
        Get the total size of the LakeShack in bytes.

        Returns:
            The total size of the LakeShack in bytes.
        """
        return sum(table_data['table_path'].stat().st_size for table_data in self._on_disk_table_info.values())
    
    def get_table_path(self, table_name: str) -> Path:
        """
        Get the file path for a given table name.

        Args:
            table_name: The name of the table.

        Returns:
            The file path of the table.
        """
        return self.path / table_name

    def write_deltalake(self, table_name: str, data: Union[pd.DataFrame, pa.Table], mode: str = 'append', **kwargs):
        """
        Write data to a specified Delta Lake table. mirrors the behavior of deltalake.write_deltalake,
        but only need to provide the table_name as an argument. If data is pd.DataFrame then convert
        to pa.Table and do not preserve index.

        Args:
            table_name: The name of the table.
            df: The DataFrame to write.
            mode: The write mode, default is 'append'.
        """
        table_path = self.get_table_path(table_name)
        data = pa.Table.from_pandas(data, preserve_index=False) if isinstance(data, pd.DataFrame) else data
        
        write_deltalake(str(table_path), data, mode=mode, **kwargs)
        
        self._update_tables()

    def sql(
        self, 
        query: str, 
        return_method: str='df', 
        use_latest_versions: bool=True
    ) -> Union[pd.DataFrame, pa.Table, np.ndarray, Any]:
        """
        Execute a SQL query using DuckDB. Convenience method for e.g. `self.con.sql(query).df()`.

        Args:
            query: The SQL query to execute.
            return_method: The return method, called on the relation `self.con.sql`. determines the
                type of object returned by the query. default is 'df' for DataFrame. options can be
                viewed here - https://duckdb.org/docs/api/python/overview.html#result-conversion.
                defaults to pandas dataframe.
            use_latest_versions: Whether to use the latest version of the Delta Lake tables. If False,
                then use the version of the table at the time of the query. default is True.

        Returns:
            The result of the query as a DataFrame.
        """
        self._update_tables()
        
        return getattr(self.con.sql(query), return_method)()
    
    def to_ibis(self, table) -> ibis.Expr:
        """
        Return ibis table object for a given table name

        Args:
            table: The name of the table.

        Returns:
            The ibis table object.
        """
        self._update_tables()

        return ibis.read_delta(self.get_table_path(table))

    def to_duckdb(self, table) -> duckdb.duckdb.DuckDBPyRelation:
        """
        Return duckdb table object for a given table name

        Args:
            table: The name of the table.
        
        Returns:
            The duckdb table object.
        """
        self._update_tables()

        return self.con.table(table)



        # try:
        #     return getattr(self.conn.sql(query), return_method)()
        # except duckdb.CatalogException as e:
        #     # possible that a new table is not registered in DuckDB yet
        #     print(f'Error: {e}')
        #     print(f'Updating SQL registry and retrying query...')
        #     self._update_tables()

        #     return getattr(self.conn.sql(query), return_method)()

        # target_table_names = np.unique(self.conn.get_table_names(query))
        # target_table_current_versions = {table_name: table.version() for table_name, table in self.tables.items()}

        # try:
        #     if use_latest_versions and (len(target_table_names) > 0 and target_table_names[0] != set()):
        #         for table_name in target_table_names:
        #             self.tables[table_name].load_with_datetime(pd.Timestamp.utcnow().isoformat(timespec='seconds'))
        #             self.conn.register(table_name, self.tables[table_name].to_pyarrow_dataset())
            
        #     self._update_tables()
        #     try:
        #         return getattr(self.conn.sql(query), return_method)()
        #     except duckdb.CatalogException as e:
        #         # possible that a new table is not registered in DuckDB yet
        #         print(f'Error: {e}')
        #         print(f'Updating SQL registry and retrying query...')
        #         self._update_tables()

        #         return getattr(self.conn.sql(query), return_method)()
            
        # finally:
        #     for table_name, version in target_table_current_versions.items():
        #         if (len(target_table_names) > 0 and target_table_names[0] != set()):
        #             self.tables[table_name].load_version(version) 
        #             self.conn.register(table_name, self.tables[table_name].to_pyarrow_dataset())


    # @property
    # def tables(self) -> Dict[str, Dict[str, Any]]:
    #     """
    #     Get a dictionary of Delta Lake tables with their metadata.

    #     Returns:
    #         A dictionary where each key is a table name and the value is a dictionary of table metadata.
    #     """
    #     table_paths = [p for p in self.path.iterdir() if p.is_dir() and (p / '_delta_log').is_dir()]
    #     return {
    #         table_path.name: {
    #             'table_name': table_path.name,
    #             'table_path': table_path,
    #             'table_path_schema': '.'.join(table_path.name.split('.')[:-1]),
    #             'table_object': DeltaTable(table_path),
    #         }
    #         for table_path in table_paths
    #     }
    