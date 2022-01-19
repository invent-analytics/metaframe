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
from typing import Any, Callable, List, Optional, Tuple, Union, Dict

from pyspark.sql import DataFrame
from pyspark.sql.group import GroupedData
from pyspark.sql.readwriter import DataFrameWriter

from warp.core.logger import LoggerMixin

LOG = LoggerMixin().log


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

    SET_PK_AFTER = {
        "groupBy": {
            "kwargs": []
        },
        "groupby": {
            "kwargs": []
        },
        "dropDuplicates": {
            "kwargs": ["subset"]
        },
        "drop_duplicates": {
            "kwargs": ["subset"]
        }
    }

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
                return self._wrapper(attr, key)
            # Non-callables are not wrapped (e.g. ``.columns``)
            return attr
        return super().__getattr__(key)

    def __getattribute__(self, key):
        if hasattr(DataFrame, key):
            raise AttributeError
        return super().__getattribute__(key)

    def __getitem__(self, key):
        return self.df.__getitem__(key)

    def _wrapper(self, func: Callable, callable_key: str) -> Callable:
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
                result = InventDataFrame(result, self.metadata)
            if callable_key in self.SET_PK_AFTER:
                result = self._set_pk_after(
                    result, callable_key, *args, **kwargs)
            return result
        return wrapped

    def _set_pk_after(
            self,
            df: DataFrame,
            callable_key: str,
            *args: Tuple[Any],
            **kwargs: Dict
    ) -> DataFrame:
        """
        Protected helper method to set primary key after applying operations
        defined in :py:attr:~`SET_PK_AFTER`. Infers the primary_key from
        *args or **kwargs

        :param df: dataframe to set primary_key on after callable is applied
        :type df: DataFrame
        :param callable_key: name of callable DataFrame method (e.g. groupBy)
        :type callable_key: str
        :param args: arguments of callable
        :type args: Tuple[Any]
        :param kwargs: keyword arguments of callable
        :type kwargs: Dict
        :return: InventDataFrame instance after primary key is set to metadata
        :rtype: InventDataFrame
        """
        # first, get from args
        if len(args) == 1 and isinstance(args[0], list):
            args = args[0]
        primary_key = set(args)
        # build from kwargs if we don't have incoming argument from args
        if not primary_key:
            for key, value in kwargs.items():
                if key in self.SET_PK_AFTER[callable_key]["kwargs"]:
                    if isinstance(value, (list, set, tuple)):
                        primary_key.update(value)
                    else:
                        primary_key.add(value)
        primary_key = list(primary_key)

        LOG.info(
            "Setting primary_key as %s after %s operation",
            primary_key, callable_key
        )

        df = df.df if isinstance(df, InventDataFrame) else df
        df = InventDataFrame(df, {**self.metadata, "primary_key": primary_key})
        return df

    def set_metadata(self, **params) -> DataFrame:
        """
        Sets metadata attribute, returns InventDataFrame with the new metadata

        :return: Object with changed metadata
        :rtype: InventDataFrame
        """
        return InventDataFrame(self.df, {**self.metadata, **params})

    @property
    def primary_key(self) -> Optional[List[str]]:
        """
        Returns primary_key information of metadata. If primary_key key does
        not exist in metadata, the return value is None

        :return: `primary_key` of :py:attr:~`metadata`
        :rtype: Optional[List[str]]
        """
        return self.metadata.get("primary_key")
