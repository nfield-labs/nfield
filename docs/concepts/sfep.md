# SFEP — why `key = value` beats nested JSON

The format tax is the accuracy lost to producing JSON *structure* — the nested
braces, commas, and quotes — rather than the field *values*. It grows with field
count and nesting depth.

FormatShield sidesteps it with the **Schema-aware Field Extraction Protocol (SFEP)**:
the model emits one flat line per field,

```
invoice.vendor = Acme Corporation
invoice.total = 1284.50
invoice.line_items[0].description = widget
```

instead of a nested object. Each line is a dot-notation path and a raw value. There
are no structural tokens to get wrong, so attention stays on the content.

## Lossless round-trip

SFEP is bijective with nested JSON: the parser splits each line on the first ` = `,
type-casts the value against the field's schema, and the assembler rebuilds the
nested object via a radix trie. Flattening then reassembling returns the original
structure — every schema field maps to exactly one path and back.

## Per-field validation

Because output is per-field, validation is per-field too: a value that violates its
constraint (wrong type, out of range, bad enum) is caught and only *that* field is
re-extracted, rather than regenerating the whole object.
