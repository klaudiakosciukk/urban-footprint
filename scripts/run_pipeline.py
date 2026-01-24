import os
import argparse
from pathlib import Path

import ee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# CONFIG

CITY_COORDS = {
    "krakow":  (19.9445, 50.0647),
    "warsaw":  (21.0122, 52.2297),
    "wroclaw": (17.0385, 51.1079),
}

# CORINE built-up/urban (Level-3) classes
CORINE_URBAN_CLASSES = [111, 112, 121, 122, 123, 124]

# Datasets (public)
EMB_COLLECTION_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
CORINE_2018_ID = "COPERNICUS/CORINE/V20/100m/2018"



# I/O utils

def ensure_dirs():
    Path("outputs/plots").mkdir(parents=True, exist_ok=True)
    Path("outputs/maps").mkdir(parents=True, exist_ok=True)
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)


def safe_remove(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass



# Geometry & data

def get_aoi(city: str, buffer_km: int) -> ee.Geometry:
    if city not in CITY_COORDS:
        raise ValueError(f"Unknown city: {city}. Available: {list(CITY_COORDS.keys())}")

    lon, lat = CITY_COORDS[city]
    return ee.Geometry.Point([lon, lat]).buffer(buffer_km * 1000)


def available_years(aoi: ee.Geometry, y0: int, y1: int):
    """Return list of years that exist in embeddings collection for this AOI."""
    col = ee.ImageCollection(EMB_COLLECTION_ID).filterBounds(aoi)
    ok = []
    for y in range(y0, y1 + 1):
        n = col.filterDate(f"{y}-01-01", f"{y+1}-01-01").size().getInfo()
        if n and n > 0:
            ok.append(y)
    return ok


def load_embedding_year(aoi: ee.Geometry, year: int) -> ee.Image:
    """Annual embeddings can come as multiple tiles -> mosaic into one image."""
    col = (ee.ImageCollection(EMB_COLLECTION_ID)
           .filterBounds(aoi)
           .filterDate(f"{year}-01-01", f"{year+1}-01-01"))
    return col.mosaic().clip(aoi)


def load_corine_urban_mask(aoi: ee.Geometry) -> ee.Image:
    corine = ee.Image(CORINE_2018_ID).select("landcover").clip(aoi)
    urban = corine.remap(CORINE_URBAN_CLASSES, [1]*len(CORINE_URBAN_CLASSES), 0).rename("urban")
    return urban


def fc_to_pandas(fc: ee.FeatureCollection) -> pd.DataFrame:
    """FeatureCollection -> pandas using getInfo (limit to <=5000 features!)."""
    features = fc.getInfo()["features"]
    rows = [f["properties"] for f in features]
    df = pd.DataFrame(rows)
    df = df.drop(columns=[".geo"], errors="ignore")
    return df



# Map export (ONLY PNG in outputs/maps)

def export_change_map_png(change_img: ee.Image, aoi: ee.Geometry, out_png: str):
    """
    Export change map as a PNG using Earth Engine thumbnail URL.
    Classes:
      0 = non-urban (black)
      1 = stable urban (yellow)
      2 = new urban (green)
    """
    import requests

    # Visualize with your required style
    vis = change_img.visualize(
        min=0,
        max=2,
        palette=["000000", "FFFF00", "00FF00"]  # black, yellow, green
    )

    # Use AOI bounds as region
    region = aoi.bounds().getInfo()["coordinates"]


    for dim in [2048, 1536, 1024, 768]:
        try:
            url = vis.getThumbURL({
                "region": region,
                "dimensions": dim,
                "format": "png"
            })

            r = requests.get(url, timeout=120)
            r.raise_for_status()

            with open(out_png, "wb") as f:
                f.write(r.content)

            if os.path.exists(out_png) and os.path.getsize(out_png) > 0:
                print(f"[OK] Saved change map PNG (dimensions={dim}): {out_png}")
                return

        except Exception as e:
            print(f"[WARN] Thumbnail export failed (dimensions={dim}): {e}")

    raise RuntimeError("Failed to export change map PNG via thumbnail URL.")



# Main pipeline

def run_pipeline(city: str, years: list[int], buffer_km: int, n_clusters: int = 10, sample_n: int = 20000):
    ensure_dirs()

    aoi = get_aoi(city, buffer_km)

    y0, y1 = years[0], years[1]
    ok_years = available_years(aoi, y0, y1)

    if not ok_years:
        raise RuntimeError(f"No embeddings available for {city} in range {y0}-{y1}.")

    if y1 not in ok_years:
        print(f"[WARN] Year {y1} not available. Using last available year: {ok_years[-1]}")
        y1 = ok_years[-1]

    years_list = [y for y in range(y0, y1 + 1) if y in ok_years]

    print(f"[INFO] City: {city} | Buffer: {buffer_km} km")
    print(f"[INFO] Years: {years_list}")

    base_year = years_list[0]
    emb_base = load_embedding_year(aoi, base_year)
    corine_urban = load_corine_urban_mask(aoi)

    # Sampling for training clusterer in EE
    band_names = emb_base.bandNames()
    sample_fc = (emb_base.addBands(corine_urban)
                 .sample(region=aoi, scale=10, numPixels=sample_n, seed=42, geometries=False))

    # Train KMeans clusterer in EE (WEKA)
    clusterer = ee.Clusterer.wekaKMeans(n_clusters).train(
        features=sample_fc,
        inputProperties=band_names
    )

    # Cluster baseline image
    clustered_base = emb_base.cluster(clusterer).rename("cluster")

    # For selecting "urban clusters" use a small sample (<=5000) due to getInfo limits
    img_with_labels = emb_base.addBands(corine_urban).addBands(clustered_base)

    sample_fc_small = img_with_labels.sample(
        region=aoi,
        scale=10,
        numPixels=min(5000, sample_n),
        seed=42,
        geometries=False
    )

    df_small = fc_to_pandas(sample_fc_small)

    if "urban" not in df_small.columns or "cluster" not in df_small.columns:
        raise RuntimeError("Missing 'urban' or 'cluster' fields in sampled dataframe.")

    urban_ratio = df_small.groupby("cluster")["urban"].mean().sort_values(ascending=False)

    # Urban clusters threshold (tunable)
    urban_clusters = urban_ratio[urban_ratio >= 0.6].index.astype(int).tolist()

    print("[INFO] Urban ratio per cluster:")
    print(urban_ratio)
    print(f"[INFO] Urban clusters (>=0.6): {urban_clusters}")

    # Compute urban area per year (km²)
    AREA_SCALE = 30  # faster/stable than 10m for area stats
    pixel_area_m2 = ee.Image.pixelArea()

    area_rows = []
    urban_masks = {}

    for y in years_list:
        emb_y = load_embedding_year(aoi, y)
        clusters_y = emb_y.cluster(clusterer).rename("cluster")

        urban_mask = clusters_y.remap(urban_clusters, [1]*len(urban_clusters), 0).rename("urban_emb")
        urban_masks[y] = urban_mask

        # robust reduceRegion (avoid timeouts)
        urban_area_m2 = (urban_mask.selfMask()
                         .multiply(pixel_area_m2)
                         .reduceRegion(
                             reducer=ee.Reducer.sum(),
                             geometry=aoi,
                             scale=AREA_SCALE,
                             maxPixels=1e13,
                             tileScale=8,
                             bestEffort=True
                         )
                         .get("urban_emb"))

        urban_km2 = ee.Number(urban_area_m2).divide(1e6).getInfo()

        area_rows.append({"year": y, "urban_km2": float(urban_km2) if urban_km2 is not None else np.nan})
        print(f"[INFO] Year {y}: urban_km2 = {urban_km2}")

    # Save table (ONLY ONE CSV)
    table_path = f"outputs/tables/urban_area_{city}.csv"
    df_area = pd.DataFrame(area_rows)
    df_area.to_csv(table_path, index=False)
    print(f"[OK] Saved table: {table_path}")

    # Plot time series (ONLY ONE PLOT)
    plt.figure(figsize=(8, 4.5))
    plt.plot(df_area["year"], df_area["urban_km2"], marker="o")
    plt.xlabel("Year")
    plt.ylabel("Urban area (km²)")
    plt.title(f"Urban footprint time series — {city} ({years_list[0]}–{years_list[-1]})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = "outputs/plots/urban_area_timeseries.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[OK] Saved plot: {plot_path}")

    # Change map (ONLY ONE PNG in outputs/maps)
    first_year = years_list[0]
    last_year = years_list[-1]

    urban0 = urban_masks[first_year]
    urban1 = urban_masks[last_year]

    # Change classes:
    # 0 = non-urban
    # 1 = stable urban
    # 2 = new urban (expansion)
    change_map = (ee.Image(0)
                  .where(urban0.eq(1).And(urban1.eq(1)), 1)
                  .where(urban0.eq(0).And(urban1.eq(1)), 2)
                  .rename("change")
                  .clip(aoi))

    out_png = f"outputs/maps/change_map_{city}_{first_year}_{last_year}.png"
    export_change_map_png(change_map, aoi, out_png)
    print(f"[OK] Saved change map PNG: {out_png}")


def parse_args():
    parser = argparse.ArgumentParser(description="Urban footprint change detection (Embeddings + CORINE baseline)")
    parser.add_argument("--city", type=str, required=True, help="krakow / warsaw / wroclaw")
    parser.add_argument("--years", type=int, nargs=2, required=True, help="start_year end_year (e.g. 2018 2022)")
    parser.add_argument("-k", "--buffer_km", type=int, default=30, help="AOI buffer radius in km")
    parser.add_argument("--clusters", type=int, default=10, help="Number of KMeans clusters (default=10)")
    parser.add_argument("--samples", type=int, default=20000, help="Number of sampled pixels for EE training (default=20000)")
    return parser.parse_args()


def main():
    args = parse_args()

    ee.Initialize()

    run_pipeline(
        city=args.city.lower(),
        years=args.years,
        buffer_km=args.buffer_km,
        n_clusters=args.clusters,
        sample_n=args.samples
    )


if __name__ == "__main__":
    main()
