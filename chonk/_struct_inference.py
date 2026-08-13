# Copyright (c) 2025 Kenneth Stott. MIT License.
# Canary: 1f431c75-0724-4729-9c8d-493751f5a62c
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Schema inference helpers for load_structured_file()."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .schema import ColumnMeta, TableMeta

if TYPE_CHECKING:
    import pyarrow as pa

# ---------------------------------------------------------------------------
# Type normalisation helpers
# ---------------------------------------------------------------------------


def _pandas_dtype_to_str(dtype: object) -> str:
    name = str(dtype)
    if name.startswith("int") or name.startswith("Int"):
        return "INTEGER"
    if name.startswith("float") or name.startswith("Float"):
        return "FLOAT"
    if name.startswith("bool"):
        return "BOOLEAN"
    if name.startswith("datetime"):
        return "TIMESTAMP"
    return "TEXT"


def _python_type_to_str(value: object) -> str:
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, int):
        return "INTEGER"
    if isinstance(value, float):
        return "FLOAT"
    if isinstance(value, (dict, list)):
        return "JSON"
    return "TEXT"


def _arrow_type_to_str(pa_type: pa.DataType) -> str:
    import pyarrow as pa

    if pa.types.is_integer(pa_type):
        return "INTEGER"
    if pa.types.is_floating(pa_type):
        return "FLOAT"
    if pa.types.is_boolean(pa_type):
        return "BOOLEAN"
    if pa.types.is_temporal(pa_type):
        return "TIMESTAMP"
    if pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type):
        return "TEXT"
    if pa.types.is_binary(pa_type) or pa.types.is_large_binary(pa_type):
        return "BINARY"
    if pa.types.is_list(pa_type) or pa.types.is_large_list(pa_type):
        return "LIST"
    if pa.types.is_struct(pa_type):
        return "STRUCT"
    return str(pa_type)


# ---------------------------------------------------------------------------
# Per-format inference
# ---------------------------------------------------------------------------


def infer_csv(data: bytes, name: str) -> TableMeta:
    try:
        import io

        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for CSV schema inference.") from exc

    df = pd.read_csv(io.BytesIO(data), nrows=100)
    row_count = sum(1 for _ in io.BytesIO(data)) - 1  # subtract header
    columns = [
        ColumnMeta(name=col, data_type=_pandas_dtype_to_str(df[col].dtype)) for col in df.columns
    ]
    return TableMeta(name=name, columns=columns, row_count=max(row_count, 0))


def infer_json(data: bytes, name: str) -> TableMeta:
    import json

    try:
        obj = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot parse JSON for {name!r}: {exc}") from exc

    if isinstance(obj, list):
        key_types: dict[str, str] = {}
        for item in obj[:100]:
            if isinstance(item, dict):
                for k, v in item.items():
                    if k not in key_types:
                        key_types[k] = _python_type_to_str(v)
        columns = [ColumnMeta(name=k, data_type=v) for k, v in key_types.items()]
        return TableMeta(name=name, columns=columns, row_count=len(obj))

    if isinstance(obj, dict):
        columns = [
            ColumnMeta(name=k, data_type=_python_type_to_str(v)) for k, v in list(obj.items())[:100]
        ]
        return TableMeta(name=name, columns=columns, row_count=1)

    return TableMeta(name=name)


def infer_jsonl(data: bytes, name: str) -> TableMeta:
    import io
    import json

    key_types: dict[str, str] = {}
    count = 0
    total = 0
    parse_failures = 0
    for raw in io.BytesIO(data):
        line = raw.strip()
        if not line:
            continue
        total += 1
        if count < 100:
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k not in key_types:
                            key_types[k] = _python_type_to_str(v)
                count += 1
            except json.JSONDecodeError:
                parse_failures += 1

    if total > 0 and count == 0:
        raise ValueError(f"All {parse_failures} non-empty lines in {name!r} failed JSON parsing")

    columns = [ColumnMeta(name=k, data_type=v) for k, v in key_types.items()]
    return TableMeta(name=name, columns=columns, row_count=total)


def infer_parquet(data: bytes, ext: str, name: str) -> TableMeta:
    try:
        import pyarrow as pa
        import pyarrow.feather as feather
        import pyarrow.ipc as ipc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for parquet/arrow/feather schema inference. "
            "Install with: pip install chonk-rag[parquet]"
        ) from exc

    buf = pa.BufferReader(data)

    if ext == ".arrow":
        reader = ipc.open_stream(buf)
        schema = reader.schema
        row_count = None
    elif ext == ".feather":
        table = feather.read_table(buf)
        schema = table.schema
        row_count = table.num_rows
    else:  # .parquet
        pf = pq.ParquetFile(buf)
        schema = pf.schema_arrow
        row_count = pf.metadata.num_rows

    columns = [
        ColumnMeta(name=field.name, data_type=_arrow_type_to_str(field.type)) for field in schema
    ]
    return TableMeta(name=name, columns=columns, row_count=row_count)
