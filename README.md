# FormatShield

**N-field structured extraction from documents with LLMs.**

Extract hundreds of structured fields from any document — without the format tax.

## Install

```bash
pip install formatshield
pip install "formatshield[groq]"  # Groq provider
```

## Quickstart

```python
from formatshield import nfield

schema = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string"},
        "total": {"type": "number"},
        "date": {"type": "string", "format": "date"},
    },
    "required": ["vendor", "total", "date"],
}

result = nfield(document_text, schema, "groq/llama-3.3-70b-versatile")
print(result.data)
```

## License

Apache 2.0
