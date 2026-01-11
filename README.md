# Urban Footprint & Expansion Mapping (2018–2025)

Detect urban change around Warsaw, Kraków, and Wrocław using annual satellite embeddings and CORINE 2018 as baseline.

## Quickstart (local)
1) Create venv and install:
   - python -m venv .venv
   - source .venv/bin/activate  (Windows: .venv\Scripts\activate)
   - pip install -r requirements.txt

2) Run:
   - python scripts/run_pipeline.py --all-cities --years 2018 2025

## Outputs
- outputs/plots/
- outputs/maps/
- outputs/tables/
