"""Extract a wide schema and read the completeness metadata.

nfield splits a wide schema into calls that fit the model, then reassembles the nested JSON.
The metadata reports how complete the result is and how many calls it took. Passing the model's
real limits lets planning use the full window. Set GROQ_API_KEY, then run this file.
"""

from nfield import NField

document = """
ACME ROBOTICS - EQUIPMENT DATASHEET
Model: AR-9 Titan
Serial: TR-2024-88213
Payload: 12 kg
Reach: 1300 mm
Axes: 6
Weight: 240 kg
Controller: AR-CTRL-7
Firmware: 5.2.1
IP rating: IP67
Warranty: 24 months
Manufacturer: Acme Robotics
Country of origin: Germany
Release year: 2024
"""

schema = {
    "type": "object",
    "properties": {
        "model": {"type": "string"},
        "serial": {"type": "string"},
        "payload_kg": {"type": "number"},
        "reach_mm": {"type": "number"},
        "axes": {"type": "integer"},
        "weight_kg": {"type": "number"},
        "controller": {"type": "string"},
        "firmware": {"type": "string"},
        "ip_rating": {"type": "string"},
        "warranty_months": {"type": "integer"},
        "manufacturer": {"type": "string"},
        "country_of_origin": {"type": "string"},
        "release_year": {"type": "integer"},
    },
}

engine = NField(
    "groq/llama-3.3-70b-versatile",
    schema,
    context_window=131_072,
    max_output_tokens=32_768,
)
result = engine.extract(document)

print(result.data)
meta = result.metadata
print(f"extracted {meta.fields_extracted}/{meta.fields_total} in {meta.K} call(s)")
