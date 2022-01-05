# Copyright (C) Invent Analytics - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
"""
================
Invent DataFrame
================

Data structure to wrap simple pyspark.sql.DataFrame to use it with metadata.

::
    from warp.spark.dataframe import InventDataFrame

    idf = InventDataFrame(df=df, metadata={"columns": ["a", "b"]})
    idf = idf.withColumn("new_col", F.lit(1))
    idf = idf.set_metadata(columns=["a", "b", "new_col"])
    idf.show()
    assert idf.metadata["columns"] == ["a", "b", "new_col"]
"""
from typing import Any, Callable, Optional, Union, Dict

from pyspark.sql import DataFrame
from pyspark.sql.group import GroupedData
from pyspark.sql.readwriter import DataFrameWriter


class InventDataFrame(DataFrame):
    """Wrapper for pyspark.sql.DataFrame with metadata.

    :param df: Wrapped object. It is usually a DataFrame but it supports
        GroupedData and DataFrameWriter as well for compatibility.
    :type df: Union[DataFrame, GroupedData, DataFrameWriter]
    :param metadata: Metadata of the dataframe.
    :type metadata: Optional[Dict]
    """

    RETURNED_CLASSES = (
        DataFrame,
        GroupedData,
        DataFrameWriter
    )

    # pylint: disable=super-init-not-called
    def __init__(
            self,
            df: Union[DataFrame, GroupedData, DataFrameWriter],
            metadata: Optional[Dict] = None
    ):
        self.df = df
        self.metadata = metadata or {}

    def __str__(self):
        return self.df.__str__() + " metadata: " + self.metadata.__str__()

    def __repr__(self):
        return self.df.__repr__() + " metadata: " + self.metadata.__str__()

    def __getattr__(self, key: str) -> Any:
        """Router between dataframe and metadata properties. If the given key
        is an attribute of the dataframe, calls it. If not, searches for the
        key in the metadata.

        """
        if hasattr(self.df, key):
            attr = getattr(self.df, key)
            if callable(attr):
                return self._wrapper(getattr(self.df, key))
            # Non-callables are not wrapped (e.g. ``.columns``)
            return getattr(self.df, key)
        return super().__getattr__(key)

    def __getattribute__(self, key):
        if hasattr(DataFrame, key):
            raise AttributeError
        return super().__getattribute__(key)

    def __getitem__(self, key):
        return self.df.__getitem__(key)

    def _wrapper(self, func: Callable) -> Callable:
        """Wrapper helper for pyspark methods. If the result of the method
        is one of the classes listed, this returns an InventDataFrame instance
        with the returned object as df attribute and the same metadata. If not,
        it simply returns the result (e.g. ``.show()``).

        :param func: Function to be wrapped
        :type func: Callable
        :return: Wrapped function
        :rtype: Callable
        """
        def wrapped(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, self.RETURNED_CLASSES):
                return InventDataFrame(result, self.metadata)
            return result
        return wrapped

    def set_metadata(self, **params) -> DataFrame:
        """
        Sets metadata attribute, returns InventDataFrame with the new metadata

        :return: Object with changed metadata
        :rtype: InventDataFrame
        """
        return InventDataFrame(self.df, {**self.metadata, **params})
