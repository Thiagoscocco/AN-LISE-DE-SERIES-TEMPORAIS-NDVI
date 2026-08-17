import glob
import os
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime, timedelta

import ee
import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon


INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"

GEE_PROJECT_ID = os.getenv("EE_PROJECT_ID", "ndvi-estudos")

DAS_STEP_DAYS = 15
MAX_CLOUD_PERCENTAGE = 95
REDUCTION_SCALE = 10
MAP_DIMENSIONS = 900

NDVI_PALETTE = [
    "#8c510a",
    "#d8b365",
    "#f6e8c3",
    "#c7eae5",
    "#5ab4ac",
    "#01665e",
]

SAFRAS_PADRONIZADAS = [
    {"safra": "2018-2019", "inicio": "2018-11-15", "fim": "2019-04-01"},
    {"safra": "2019-2020", "inicio": "2019-11-12", "fim": "2020-03-31"},
    {"safra": "2021-2022", "inicio": "2021-11-10", "fim": "2022-04-06"},
    {"safra": "2022-2023", "inicio": "2022-11-01", "fim": "2023-04-22"},
    {"safra": "2023-2024", "inicio": "2023-11-14", "fim": "2024-04-02"},
    {"safra": "2025-2026", "inicio": "2025-11-11", "fim": "2026-03-31"},
]


def log(message: str):
    print(f"[DAS] {message}", flush=True)


def validate_configuration():
    if GEE_PROJECT_ID == "SEU_PROJECT_ID":
        raise ValueError(
            "Defina GEE_PROJECT_ID no topo do script ou exporte EE_PROJECT_ID no terminal."
        )
    if DAS_STEP_DAYS < 1:
        raise ValueError("DAS_STEP_DAYS deve ser maior que zero.")
    if not SAFRAS_PADRONIZADAS:
        raise ValueError("Adicione ao menos uma safra em SAFRAS_PADRONIZADAS.")

    for safra in SAFRAS_PADRONIZADAS:
        for key in ("safra", "inicio", "fim"):
            if key not in safra:
                raise ValueError(f"Cada safra deve conter '{key}'.")
        inicio = datetime.strptime(safra["inicio"], "%Y-%m-%d").date()
        fim = datetime.strptime(safra["fim"], "%Y-%m-%d").date()
        if fim < inicio:
            raise ValueError(f"Data final anterior ao inicio na safra {safra['safra']}.")


def initialize_earth_engine():
    try:
        ee.Initialize(project=GEE_PROJECT_ID)
    except Exception:
        log("Autenticacao do Earth Engine necessaria. Iniciando login...")
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT_ID)


def extract_geometry_from_kmz(input_dir: str):
    kmz_files = glob.glob(os.path.join(input_dir, "*.kmz"))
    if not kmz_files:
        raise FileNotFoundError("Nenhum arquivo .kmz encontrado na pasta inputs.")

    kmz_path = kmz_files[0]
    temp_kml_dir = os.path.join(input_dir, "temp_kml")
    os.makedirs(temp_kml_dir, exist_ok=True)

    with zipfile.ZipFile(kmz_path, "r") as zip_ref:
        zip_ref.extractall(temp_kml_dir)

    kml_files = glob.glob(os.path.join(temp_kml_dir, "*.kml"))
    if not kml_files:
        raise FileNotFoundError("Nenhum arquivo .kml encontrado dentro do KMZ.")

    fiona.drvsupport.supported_drivers["KML"] = "rw"
    gdf = gpd.read_file(kml_files[0], driver="KML")
    if gdf.empty:
        raise ValueError("O arquivo KML nao possui geometrias validas.")

    return gdf.to_crs("EPSG:4326").geometry.union_all()


def ring_to_xy_coords(ring):
    return [[coord[0], coord[1]] for coord in ring.coords]


def polygon_to_ee_geometry(polygon: Polygon):
    shell = ring_to_xy_coords(polygon.exterior)
    holes = [ring_to_xy_coords(interior) for interior in polygon.interiors]
    return ee.Geometry.Polygon([shell, *holes], proj="EPSG:4326", geodesic=False)


def shapely_to_ee_geometry(geometry):
    if not geometry.is_valid:
        geometry = geometry.buffer(0)

    if isinstance(geometry, Polygon):
        return polygon_to_ee_geometry(geometry)

    if isinstance(geometry, MultiPolygon):
        polygons = []
        for polygon in geometry.geoms:
            shell = ring_to_xy_coords(polygon.exterior)
            holes = [ring_to_xy_coords(interior) for interior in polygon.interiors]
            polygons.append([shell, *holes])
        return ee.Geometry.MultiPolygon(polygons, proj="EPSG:4326", geodesic=False)

    raise ValueError(f"Geometria nao suportada: {geometry.geom_type}")


def prepare_s2_image(image):
    scl = image.select("SCL")
    qa60 = image.select("QA60")

    valid_scl = (
        scl.neq(0)
        .And(scl.neq(1))
        .And(scl.neq(3))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    valid_qa60 = qa60.bitwiseAnd(1 << 10).eq(0).And(qa60.bitwiseAnd(1 << 11).eq(0))
    mask = valid_scl.And(valid_qa60)

    reflectance = image.select(["B4", "B8"], ["RED", "NIR"]).multiply(0.0001)
    ndvi = reflectance.normalizedDifference(["NIR", "RED"]).rename("NDVI")
    return ndvi.updateMask(mask).copyProperties(image, ["system:time_start"])


def build_sentinel_collection(ee_geometry, start_date: date, end_date: date):
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(ee_geometry)
        .filterDate(start_date.isoformat(), (end_date + timedelta(days=1)).isoformat())
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_PERCENTAGE))
        .map(prepare_s2_image)
    )


def compute_collection_stats(collection, ee_geometry):
    image_count = int(collection.size().getInfo())
    if image_count == 0:
        return {"status": "no_scenes", "image_count": 0, "pixel_count": 0}

    ndvi_image = collection.median()
    stats = ndvi_image.reduceRegion(
        reducer=ee.Reducer.median().combine(
            reducer2=ee.Reducer.count(),
            sharedInputs=True,
        ),
        geometry=ee_geometry,
        scale=REDUCTION_SCALE,
        maxPixels=1e9,
        bestEffort=True,
    ).getInfo()

    if not stats or stats.get("NDVI_median") is None:
        return {
            "status": "no_valid_pixels",
            "image": ndvi_image,
            "image_count": image_count,
            "pixel_count": 0,
        }

    return {
        "status": "ok",
        "image": ndvi_image,
        "image_count": image_count,
        "ndvi_median": float(stats["NDVI_median"]),
        "pixel_count": int(stats.get("NDVI_count", 0) or 0),
    }


def build_das_periods(inicio: date, fim: date):
    periods = []
    offset_days = 0
    current_start = inicio

    while current_start <= fim:
        current_end = min(current_start + timedelta(days=DAS_STEP_DAYS - 1), fim)
        periods.append(
            {
                "das": offset_days,
                "inicio": current_start,
                "fim": current_end,
            }
        )
        offset_days += DAS_STEP_DAYS
        current_start = current_end + timedelta(days=1)

    return periods


def percent(value: int, total: int):
    if total == 0:
        return 100.0
    return (value / total) * 100


def make_placeholder_image(label: str, output_path: str):
    plt.figure(figsize=(6, 6))
    plt.text(0.5, 0.55, "Sem dados validos de NDVI", ha="center", va="center", fontsize=14)
    plt.text(0.5, 0.43, label, ha="center", va="center", fontsize=12, color="#555555")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def download_ndvi_image(ndvi_image, ee_geometry, geometry_bounds, label: str, output_path: str):
    if ndvi_image is None:
        make_placeholder_image(label, output_path)
        return

    border = ee.Image().byte().paint(featureCollection=ee_geometry, color=1, width=2).visualize(
        palette=["#000000"]
    )
    visualized = ndvi_image.clip(ee_geometry).visualize(
        min=0,
        max=1,
        palette=NDVI_PALETTE,
    ).blend(border)

    minx, miny, maxx, maxy = geometry_bounds
    region = [[minx, maxy], [maxx, maxy], [maxx, miny], [minx, miny], [minx, maxy]]

    url = visualized.getThumbURL(
        {
            "region": region,
            "dimensions": MAP_DIMENSIONS,
            "format": "png",
        }
    )

    try:
        urllib.request.urlretrieve(url, output_path)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        make_placeholder_image(f"{label} - falha no download", output_path)


def plot_single_harvest(df: pd.DataFrame, safra: str, output_path: str):
    plt.figure(figsize=(12, 5))
    plt.plot(df["DAS"], df["NDVI"], color="forestgreen", linewidth=2, marker="o", markersize=5)
    plt.title(f"Curva de NDVI por DAS - Safra {safra}")
    plt.xlabel("DAS")
    plt.ylabel("NDVI mediano do talhao")
    plt.ylim(0, 1)
    plt.xticks(df["DAS"])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_overlay(all_data: pd.DataFrame, output_path: str):
    plt.figure(figsize=(13, 6))
    for safra, df_safra in all_data.groupby("Safra"):
        plt.plot(
            df_safra["DAS"],
            df_safra["NDVI"],
            linewidth=2,
            marker="o",
            markersize=4,
            label=safra,
        )

    plt.title("Curvas de NDVI por DAS - Safras sobrepostas")
    plt.xlabel("DAS")
    plt.ylabel("NDVI mediano do talhao")
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_single_harvest(safra_config, ee_geometry, geometry_bounds, global_index_start, total_periods):
    safra = safra_config["safra"]
    inicio = datetime.strptime(safra_config["inicio"], "%Y-%m-%d").date()
    fim = datetime.strptime(safra_config["fim"], "%Y-%m-%d").date()
    periods = build_das_periods(inicio, fim)

    harvest_dir = os.path.join(OUTPUT_DIR, safra)
    maps_dir = os.path.join(harvest_dir, "mapas_das")
    os.makedirs(maps_dir, exist_ok=True)

    rows = []
    for local_index, period in enumerate(periods, start=1):
        das_label = f"DAS_{period['das']:03d}"
        collection = build_sentinel_collection(ee_geometry, period["inicio"], period["fim"])
        stats = compute_collection_stats(collection, ee_geometry)

        global_done = global_index_start + local_index
        progress_text = f"{percent(global_done, total_periods):.1f}%"

        if stats["status"] == "ok":
            ndvi_value = stats["ndvi_median"]
            status_text = f"NDVI={ndvi_value:.4f}"
        elif stats["status"] == "no_scenes":
            ndvi_value = None
            status_text = "sem cenas"
        else:
            ndvi_value = None
            status_text = "sem pixel valido"

        log(f"{safra} | {das_label} | progresso {progress_text} | {status_text}")

        rows.append(
            {
                "Safra": safra,
                "DAS": period["das"],
                "Rotulo_DAS": das_label,
                "Data_Inicial": period["inicio"].isoformat(),
                "Data_Final": period["fim"].isoformat(),
                "NDVI": ndvi_value,
                "Numero_Imagens": stats["image_count"],
                "Pixels_Validos": stats["pixel_count"],
            }
        )

        image_path = os.path.join(maps_dir, f"ndvi_{das_label}.png")
        image_label = f"{safra} - {das_label}"
        download_ndvi_image(
            stats.get("image"),
            ee_geometry,
            geometry_bounds,
            image_label,
            image_path,
        )

    df = pd.DataFrame(rows)
    df["Data_Inicial"] = pd.to_datetime(df["Data_Inicial"])
    df["Data_Final"] = pd.to_datetime(df["Data_Final"])

    csv_path = os.path.join(harvest_dir, f"ndvi_das_{safra.replace('-', '_')}.csv")
    graph_path = os.path.join(harvest_dir, f"curva_ndvi_das_{safra.replace('-', '_')}.png")

    df.to_csv(csv_path, index=False)
    plot_single_harvest(df, safra, graph_path)

    return df


def main():
    validate_configuration()
    initialize_earth_engine()

    log("Iniciando processamento DAS...")
    geometry = extract_geometry_from_kmz(INPUT_DIR)
    ee_geometry = shapely_to_ee_geometry(geometry)
    geometry_bounds = geometry.bounds
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_periods = 0
    for safra_config in SAFRAS_PADRONIZADAS:
        inicio = datetime.strptime(safra_config["inicio"], "%Y-%m-%d").date()
        fim = datetime.strptime(safra_config["fim"], "%Y-%m-%d").date()
        total_periods += len(build_das_periods(inicio, fim))

    all_frames = []
    global_index_start = 0
    for safra_config in SAFRAS_PADRONIZADAS:
        inicio = datetime.strptime(safra_config["inicio"], "%Y-%m-%d").date()
        fim = datetime.strptime(safra_config["fim"], "%Y-%m-%d").date()
        period_count = len(build_das_periods(inicio, fim))
        all_frames.append(
            run_single_harvest(
                safra_config,
                ee_geometry,
                geometry_bounds,
                global_index_start,
                total_periods,
            )
        )
        global_index_start += period_count

    consolidated = pd.concat(all_frames, ignore_index=True)
    consolidated_path = os.path.join(OUTPUT_DIR, "ndvi_das_consolidado.csv")
    overlay_path = os.path.join(OUTPUT_DIR, "curvas_ndvi_das_safras_sobrepostas.png")

    consolidated.to_csv(consolidated_path, index=False)
    plot_overlay(consolidated, overlay_path)

    log("100.0% concluido. Resultados salvos em 'outputs'.")


if __name__ == "__main__":
    main()
