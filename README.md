# polars-analysis

## Fix

Use:

```python
results.show_graph()
```

Not:

```python
results.show_graph*()
```

## Why the error happens

`show_graph` is a method on a Polars `LazyFrame`.

When Python sees `results.show_graph*()`, it parses that as:

```python
results.show_graph * ()
```

That is an attempt to multiply a method object by a tuple, which produces:

```python
TypeError: can't multiply sequence by non-int of type 'method'
```

If you want to inspect the lazy query plan, call the method normally:

```python
results.show_graph()
```
