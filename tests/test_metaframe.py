"""
Unit tests
"""

import unittest

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from metaframe import MetaFrame


class TestMetaFrame(unittest.TestCase):
    """MetaFrame unit tests"""

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.master("local[4]").getOrCreate()

    def test_invent_dataframe(self):
        """
        Test MetaFrame use cases
        """
        df = self.spark.createDataFrame(
            [
                ("p1", 6, "2019-12-31"),
                ("p2", 4, "2019-12-31"),
                ("p3", 10, "2019-12-31"),
                ("p4", 4, "2019-12-31"),
                ("p5", 3, "2019-12-31"),
                ("p6", 7, "2019-12-31"),
                ("p7", 18, "2019-12-31"),
                ("p8", 44, "2019-12-31"),
                ("p1", 6, "2020-01-01"),
                ("p2", 4, "2020-01-01"),
                ("p3", 10, "2020-01-01"),
                ("p4", 4, "2020-01-01"),
                ("p5", 3, "2020-01-01"),
                ("p6", 7, "2020-01-01"),
                ("p7", 18, "2020-01-01"),
                ("p8", 44, "2020-01-01"),
                ("p1", 16, "2020-01-02"),
                ("p2", 4, "2020-01-02"),
                ("p3", 3, "2020-01-02"),
                ("p4", 6, "2020-01-02"),
                ("p5", 7, "2020-01-02"),
                ("p6", 7, "2020-01-02"),
            ],
            ["product_id", "quantity", "date"],
        )
        metadata = {"foo": "bar"}
        # Creation
        df = MetaFrame(df, metadata)
        self.assertIsInstance(df.df, DataFrame)
        self.assertDictEqual(df.metadata, metadata)
        # Non callables
        self.assertListEqual(df.columns, ["product_id", "quantity", "date"])
        self.assertIsInstance(df, MetaFrame)
        self.assertDictEqual(df.metadata, metadata)
        # Get item
        self.assertEqual(str(df["product_id"]), str(F.col("product_id")))
        self.assertIsInstance(df, MetaFrame)
        self.assertDictEqual(df.metadata, metadata)
        # Callables with no MetaFrame returns
        df.show()
        self.assertIsInstance(df, MetaFrame)
        self.assertDictEqual(df.metadata, metadata)
        # Callabes with MetaFrame returns
        df = df.withColumn("new_col", F.lit(0))
        self.assertIsInstance(df, MetaFrame)
        self.assertDictEqual(df.metadata, metadata)
        self.assertListEqual(df.columns, ["product_id", "quantity", "date", "new_col"])
        # Selecting
        df = df.select("product_id", "new_col")
        self.assertDictEqual(df.metadata, metadata)
        self.assertListEqual(df.columns, ["product_id", "new_col"])
        # Set metadata
        df = df.set_metadata(foo="baz")
        self.assertDictEqual(df.metadata, {"foo": "baz"})

    def test__set_pk_after(self):
        """
        test _set_pk_after method of MetaFrame that automatically sets
        primary key information after groupBy, dropDuplicates and distinct
        """
        df = self.spark.createDataFrame(
            [
                ("p1", 6, "2019-12-31"),
                ("p2", 4, "2019-12-31"),
                ("p3", 10, "2019-12-31"),
                ("p4", 4, "2019-12-31"),
                ("p5", 3, "2019-12-31"),
                ("p6", 7, "2019-12-31"),
                ("p7", 18, "2019-12-31"),
                ("p8", 44, "2019-12-31"),
                ("p1", 6, "2020-01-01"),
                ("p2", 4, "2020-01-01"),
                ("p3", 10, "2020-01-01"),
                ("p4", 4, "2020-01-01"),
                ("p5", 3, "2020-01-01"),
                ("p6", 7, "2020-01-01"),
                ("p7", 18, "2020-01-01"),
                ("p8", 44, "2020-01-01"),
                ("p1", 16, "2020-01-02"),
                ("p2", 4, "2020-01-02"),
                ("p3", 3, "2020-01-02"),
                ("p4", 6, "2020-01-02"),
                ("p5", 7, "2020-01-02"),
                ("p6", 7, "2020-01-02"),
            ],
            ["product_id", "quantity", "date"],
        )
        metadata = {"foo": "bar"}
        df = MetaFrame(df, metadata)

        # initially pk is empty
        self.assertIsNone(df.primary_key)

        # test groupBy & groupby
        df_date: MetaFrame = df.groupBy("date").agg(F.sum("quantity").alias("quantity"))
        self.assertEqual(df_date.primary_key, ["date"])

        df_date: MetaFrame = df.groupby("date").agg(F.sum("quantity").alias("quantity"))
        self.assertEqual(df_date.primary_key, ["date"])

        # test dropDuplicates
        df_products: MetaFrame = df.dropDuplicates(["product_id"])
        self.assertEqual(df_products.primary_key, ["product_id"])

        # test dropDuplicates with kwarg
        df_products: MetaFrame = df.dropDuplicates(subset=["product_id"])
        self.assertEqual(df_products.primary_key, ["product_id"])

        # test dropDuplicates without arg
        df_products: MetaFrame = df.select("product_id").dropDuplicates()
        self.assertEqual(df_products.primary_key, ["product_id"])

        # test drop_duplicates
        df_products: MetaFrame = df.drop_duplicates(["product_id"])
        self.assertEqual(df_products.primary_key, ["product_id"])

        # test drop_duplicates with kwarg
        df_products: MetaFrame = df.drop_duplicates(subset=["product_id"])
        self.assertEqual(df_products.primary_key, ["product_id"])

        # test drop_duplicates without arg
        df_products: MetaFrame = df.select("product_id").drop_duplicates()
        self.assertEqual(df_products.primary_key, ["product_id"])

        # test distinct
        df_products: MetaFrame = df.select("product_id").distinct()
        self.assertEqual(df_products.primary_key, ["product_id"])
