# MetaFrame

A data structure that extends ``pyspark.sql.DataFrame`` with metadata information.

## Usage

```python
from metaframe import MetaFrame
mf = MetaFrame(df=df, metadata={"columns": ["a", "b"]})
mf = mf.withColumn("new_col", F.lit(1))
mf = mf.set_metadata(columns=["a", "b", "new_col"])
mf.show()
assert mf.metadata["columns"] == ["a", "b", "new_col"]
```
