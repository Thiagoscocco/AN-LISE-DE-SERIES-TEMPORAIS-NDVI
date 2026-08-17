import glob
import math
import os
import textwrap
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta

import ee
import fiona
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from shapely.geometry import MultiPolygon, Polygon


INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"

GEE_PROJECT_ID = os.getenv("EE_PROJECT_ID", "ndvi-estudos")

# 2017 gera a safra 2017/2018
HARVEST_START_YEAR = 2017
HARVEST_END_YEAR = 2025

# Outubro a Maio
START_MONTH = 10
END_MONTH = 5

# Resolucao temporal da curva
PERIOD_STEP_DAYS = 14
LONG_GAP_THRESHOLD_DAYS = 21

MAX_CLOUD_PERCENTAGE = 95
REDUCTION_SCALE = 10
MAP_DIMENSIONS = 900

# Regras da classificacao
APTA_MIN_COVERAGE = 75.0
APTA_MAX_GAPS_3W = 1
APTA_MAX_SPATIAL_STD = 0.12
APTA_MIN_AMPLITUDE = 0.18

RESSALVA_MIN_COVERAGE = 55.0
RESSALVA_MAX_GAPS_3W = 2
RESSALVA_MAX_SPATIAL_STD = 0.18
RESSALVA_MIN_AMPLITUDE = 0.12

NDVI_PALETTE = [
    "#8c510a",
    "#d8b365",
    "#f6e8c3",
    "#c7eae5",
    "#5ab4ac",
    "#01665e",
]


@dataclass
class HarvestAssessment:
    classification: str
    coverage_percent: float
    total_weeks: int
    observed_weeks: int
    three_week_gap_count: int
    amplitude: float
    median_spatial_std: float | None
    note: str


def log(message: str):
    print(f"[NDVI] {message}", flush=True)


def validate_configuration():
    if GEE_PROJECT_ID == "SEU_PROJECT_ID":
        raise ValueError(
            "Defina GEE_PROJECT_ID no topo do script ou exporte EE_PROJECT_ID no terminal."
        )
    if HARVEST_START_YEAR > HARVEST_END_YEAR:
        raise ValueError("HARVEST_START_YEAR nao pode ser maior que HARVEST_END_YEAR.")
    if HARVEST_START_YEAR < 2017:
        raise ValueError(
            "Sentinel-2 no Earth Engine permite safras completas de outubro a maio a partir de 2017/2018."
        )
    if not 1 <= START_MONTH <= 12 or not 1 <= END_MONTH <= 12:
        raise ValueError("START_MONTH e END_MONTH devem estar entre 1 e 12.")
    if PERIOD_STEP_DAYS < 1:
        raise ValueError("PERIOD_STEP_DAYS deve ser maior que zero.")


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


def harvest_start_date(year_start: int) -> date:
    return date(year_start, START_MONTH, 1)


def harvest_end_date(year_start: int) -> date:
    year_end = year_start + 1
    next_month = END_MONTH + 1
    if next_month == 13:
        return date(year_end + 1, 1, 1) - timedelta(days=1)
    return date(year_end, next_month, 1) - timedelta(days=1)


def time_windows(year_start: int):
    current = harvest_start_date(year_start)
    end = harvest_end_date(year_start)

    windows = []
    while current <= end:
        period_end = min(current + timedelta(days=PERIOD_STEP_DAYS - 1), end)
        windows.append((current, period_end))
        current = period_end + timedelta(days=1)
    return windows


def harvest_months(year_start: int):
    months = []
    current_year = year_start
    current_month = START_MONTH
    while True:
        months.append((current_year, current_month))
        if current_month == END_MONTH and current_year == year_start + 1:
            break
        current_month += 1
        if current_month == 13:
            current_month = 1
            current_year += 1
    return months


def month_date_range(year: int, month: int):
    start = date(year, month, 1)
    if month == 12:
        end = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end = date(year, month + 1, 1) - timedelta(days=1)
    return start, end


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


def build_sentinel_collection(ee_geometry, start_date_str: str, end_date_exclusive_str: str):
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(ee_geometry)
        .filterDate(start_date_str, end_date_exclusive_str)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_PERCENTAGE))
        .map(prepare_s2_image)
    )


def compute_collection_stats(collection, ee_geometry):
    image_count = int(collection.size().getInfo())
    if image_count == 0:
        return {
            "status": "no_scenes",
            "image_count": 0,
            "pixel_count": 0,
        }

    ndvi_image = collection.median()
    stats = ndvi_image.reduceRegion(
        reducer=ee.Reducer.median()
        .combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.count(), sharedInputs=True),
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
        "ndvi_stddev": None if stats.get("NDVI_stdDev") is None else float(stats["NDVI_stdDev"]),
        "pixel_count": None if stats.get("NDVI_count") is None else int(stats["NDVI_count"]),
    }


def compute_period_ndvi(ee_geometry, period_start: date, period_end: date):
    start_date_str = period_start.strftime("%Y-%m-%d")
    end_date_exclusive_str = (period_end + timedelta(days=1)).strftime("%Y-%m-%d")

    log(
        f"Consultando periodo {period_start.strftime('%d/%m/%Y')} a {period_end.strftime('%d/%m/%Y')}..."
    )
    collection = build_sentinel_collection(ee_geometry, start_date_str, end_date_exclusive_str)
    stats = compute_collection_stats(collection, ee_geometry)

    if stats["status"] == "no_scenes":
        log("Periodo sem cenas Sentinel-2 para o talhao.")
        return {
            "ndvi": None,
            "image_count": 0,
            "pixel_count": 0,
        }

    if stats["status"] == "no_valid_pixels":
        log(
            f"Periodo com {stats['image_count']} imagem(ns), mas sem pixel valido apos mascara."
        )
        return {
            "ndvi": None,
            "image_count": stats["image_count"],
            "pixel_count": 0,
        }

    log(
        f"NDVI do periodo: {stats['ndvi_median']:.4f} com {stats['image_count']} imagem(ns) e "
        f"{stats['pixel_count']} pixel(s) validos."
    )
    return {
        "ndvi": stats["ndvi_median"],
        "image_count": stats["image_count"],
        "pixel_count": stats["pixel_count"] or 0,
    }


def build_harvest_dataframe(ee_geometry, year_start: int):
    rows = []
    for period_start, period_end in time_windows(year_start):
        period_stats = compute_period_ndvi(ee_geometry, period_start, period_end)
        rows.append(
            {
                "Data_Inicial": period_start.isoformat(),
                "Data_Final": period_end.isoformat(),
                "Data_Referencia": period_start.isoformat(),
                "Rotulo": period_start.strftime("%d/%m/%Y"),
                "NDVI": period_stats["ndvi"],
                "Numero_Imagens": period_stats["image_count"],
                "Pixels_Validos": period_stats["pixel_count"],
                "Tem_Observacao": period_stats["ndvi"] is not None,
            }
        )

    df = pd.DataFrame(rows)
    df["Data_Referencia"] = pd.to_datetime(df["Data_Referencia"])
    return df


def count_long_gaps(ndvi_series: pd.Series):
    is_missing = ndvi_series.isna().tolist()
    gap_count = 0
    current_gap = 0

    for missing in is_missing:
        if missing:
            current_gap += 1
            continue
        if current_gap * PERIOD_STEP_DAYS >= LONG_GAP_THRESHOLD_DAYS:
            gap_count += 1
        current_gap = 0

    if current_gap * PERIOD_STEP_DAYS >= LONG_GAP_THRESHOLD_DAYS:
        gap_count += 1

    return gap_count


def compute_monthly_uniformity(ee_geometry, year: int, month: int):
    start, end = month_date_range(year, month)
    start_date_str = start.strftime("%Y-%m-%d")
    end_date_exclusive_str = (end + timedelta(days=1)).strftime("%Y-%m-%d")
    collection = build_sentinel_collection(ee_geometry, start_date_str, end_date_exclusive_str)
    stats = compute_collection_stats(collection, ee_geometry)

    if stats["status"] != "ok":
        return {
            "label": f"{month:02d}/{year}",
            "ndvi_median": None,
            "spatial_stddev": None,
            "image_count": stats["image_count"],
            "image": None,
        }

    return {
        "label": f"{month:02d}/{year}",
        "ndvi_median": stats["ndvi_median"],
        "spatial_stddev": stats["ndvi_stddev"],
        "image_count": stats["image_count"],
        "image": stats["image"],
    }


def assess_harvest(df: pd.DataFrame, monthly_stats: list[dict]):
    total_weeks = len(df)
    observed_weeks = int(df["Tem_Observacao"].sum())
    coverage_percent = (observed_weeks / total_weeks) * 100 if total_weeks else 0.0
    three_week_gap_count = count_long_gaps(df["NDVI"])

    observed_ndvi = df["NDVI"].dropna()
    amplitude = 0.0
    if not observed_ndvi.empty:
        amplitude = float(observed_ndvi.max() - observed_ndvi.min())

    spatial_std_values = [
        month["spatial_stddev"] for month in monthly_stats if month["spatial_stddev"] is not None
    ]
    median_spatial_std = None
    if spatial_std_values:
        median_spatial_std = float(pd.Series(spatial_std_values).median())

    if (
        coverage_percent >= APTA_MIN_COVERAGE
        and three_week_gap_count <= APTA_MAX_GAPS_3W
        and amplitude >= APTA_MIN_AMPLITUDE
        and (median_spatial_std is None or median_spatial_std <= APTA_MAX_SPATIAL_STD)
    ):
        classification = "Apta"
        note = "Boa cobertura temporal, curva utilizavel e uniformidade espacial consistente."
    elif (
        coverage_percent >= RESSALVA_MIN_COVERAGE
        and three_week_gap_count <= RESSALVA_MAX_GAPS_3W
        and amplitude >= RESSALVA_MIN_AMPLITUDE
        and (median_spatial_std is None or median_spatial_std <= RESSALVA_MAX_SPATIAL_STD)
    ):
        classification = "Apta com ressalvas"
        note = "Curva ainda utilizavel, mas com limitacoes de cobertura, ruido ou uniformidade."
    else:
        classification = "Inapta"
        note = "Cobertura temporal ou consistencia espacial insuficiente para uma safra confiavel."

    return HarvestAssessment(
        classification=classification,
        coverage_percent=coverage_percent,
        total_weeks=total_weeks,
        observed_weeks=observed_weeks,
        three_week_gap_count=three_week_gap_count,
        amplitude=amplitude,
        median_spatial_std=median_spatial_std,
        note=note,
    )


def harvest_folder_name(year_start: int):
    return f"{year_start}-{year_start + 1}"


def render_ndvi_plot(df: pd.DataFrame, assessment: HarvestAssessment, year_start: int, output_path: str):
    plt.figure(figsize=(13, 6))
    plt.plot(
        df["Data_Referencia"],
        df["NDVI"],
        color="forestgreen",
        linewidth=2,
        marker="o",
        markersize=5,
    )

    plt.title(f"Curva de NDVI em janelas de 14 dias - Safra {year_start}/{year_start + 1}")
    plt.xlabel("Periodo")
    plt.ylabel("NDVI mediano do talhao")
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.6)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    plt.xticks(rotation=45)

    metrics_text = (
        f"Lacunas >=3 semanas: {assessment.three_week_gap_count}\n"
        f"Cobertura temporal: {assessment.coverage_percent:.1f}%\n"
        f"Periodos avaliados: {assessment.total_weeks}"
    )
    plt.gcf().text(
        0.015,
        0.93,
        metrics_text,
        fontsize=10,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#777777"},
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def make_month_placeholder(label: str, output_path: str):
    plt.figure(figsize=(6, 6))
    plt.text(0.5, 0.55, "Sem dados validos de NDVI", ha="center", va="center", fontsize=14)
    plt.text(0.5, 0.43, label, ha="center", va="center", fontsize=12, color="#555555")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def download_monthly_ndvi_image(
    ndvi_image,
    ee_geometry,
    geometry_bounds,
    label: str,
    output_path: str,
):
    if ndvi_image is None:
        make_month_placeholder(label, output_path)
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
        make_month_placeholder(f"{label} - falha no download", output_path)


def save_monthly_maps(
    ee_geometry,
    geometry_bounds,
    year_start: int,
    harvest_dir: str,
):
    monthly_dir = os.path.join(harvest_dir, "mapas_ndvi")
    os.makedirs(monthly_dir, exist_ok=True)

    monthly_stats = []
    for year, month in harvest_months(year_start):
        label = f"{month:02d}_{year}"
        log(f"Gerando mapa mensal de NDVI para {month:02d}/{year}...")
        month_stats = compute_monthly_uniformity(ee_geometry, year, month)
        monthly_stats.append(month_stats)
        output_path = os.path.join(monthly_dir, f"ndvi_{label}.png")
        download_monthly_ndvi_image(
            month_stats["image"],
            ee_geometry,
            geometry_bounds,
            month_stats["label"],
            output_path,
        )

    return monthly_stats


def build_assessment_text(assessment: HarvestAssessment, monthly_stats: list[dict]):
    uniformity_line = "Uniformidade espacial mensal indisponivel."
    if assessment.median_spatial_std is not None:
        uniformity_line = (
            f"Mediana do desvio padrao espacial mensal do NDVI: "
            f"{assessment.median_spatial_std:.3f}."
        )

    month_lines = []
    for month in monthly_stats:
        month_median = "-" if month["ndvi_median"] is None else f"{month['ndvi_median']:.3f}"
        month_std = "-" if month["spatial_stddev"] is None else f"{month['spatial_stddev']:.3f}"
        month_lines.append(
            f"{month['label']}: mediana={month_median} | desvio_padrao_espacial={month_std} | "
            f"imagens={month['image_count']}"
        )

    return [
        f"Classificacao: {assessment.classification}",
        f"Cobertura temporal: {assessment.coverage_percent:.1f}% "
        f"({assessment.observed_weeks}/{assessment.total_weeks} periodos com dado).",
        f"Lacunas de 3 semanas ou mais: {assessment.three_week_gap_count}.",
        f"Amplitude observada da curva NDVI: {assessment.amplitude:.3f}.",
        uniformity_line,
        assessment.note,
        "",
        "Resumo mensal para avaliar a uniformidade:",
        *month_lines,
    ]


def save_simple_pdf(title: str, lines: list[str], output_path: str):
    wrapped_lines = []
    for line in lines:
        if not line:
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(textwrap.wrap(line, width=95))

    with PdfPages(output_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        plt.axis("off")

        y = 0.97
        plt.text(0.05, y, title, fontsize=16, fontweight="bold", ha="left", va="top")
        y -= 0.045

        for line in wrapped_lines:
            if y < 0.05:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                fig.patch.set_facecolor("white")
                plt.axis("off")
                y = 0.97
            plt.text(0.05, y, line, fontsize=11, ha="left", va="top")
            y -= 0.025 if line else 0.018

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def save_harvest_outputs(
    df: pd.DataFrame,
    ee_geometry,
    geometry_bounds,
    year_start: int,
):
    harvest_dir = os.path.join(OUTPUT_DIR, harvest_folder_name(year_start))
    os.makedirs(harvest_dir, exist_ok=True)

    csv_path = os.path.join(harvest_dir, f"ndvi_14dias_{year_start}_{year_start + 1}.csv")
    graph_path = os.path.join(harvest_dir, f"curva_ndvi_{year_start}_{year_start + 1}.png")
    pdf_path = os.path.join(harvest_dir, f"avaliacao_{year_start}_{year_start + 1}.pdf")

    df.to_csv(csv_path, index=False)

    monthly_stats = save_monthly_maps(ee_geometry, geometry_bounds, year_start, harvest_dir)
    assessment = assess_harvest(df, monthly_stats)

    render_ndvi_plot(df, assessment, year_start, graph_path)
    save_simple_pdf(
        f"Avaliacao da safra {year_start}/{year_start + 1}",
        build_assessment_text(assessment, monthly_stats),
        pdf_path,
    )

    return {
        "safra": f"{year_start}/{year_start + 1}",
        "classificacao": assessment.classification,
        "cobertura_temporal": assessment.coverage_percent,
        "semanas_totais": assessment.total_weeks,
        "semanas_observadas": assessment.observed_weeks,
        "lacunas_3_semanas": assessment.three_week_gap_count,
        "amplitude": assessment.amplitude,
        "desvio_padrao_espacial_mediano": assessment.median_spatial_std,
        "pasta_saida": harvest_dir,
    }


def save_final_summary(summary_rows: list[dict]):
    if not summary_rows:
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUTPUT_DIR, "resumo_safras.csv")
    summary_df.to_csv(summary_csv, index=False)

    aptas = summary_df[summary_df["classificacao"] == "Apta"]["safra"].tolist()
    ressalvas = summary_df[summary_df["classificacao"] == "Apta com ressalvas"]["safra"].tolist()
    inaptas = summary_df[summary_df["classificacao"] == "Inapta"]["safra"].tolist()

    def format_group(items):
        return ", ".join(items) if items else "Nenhuma"

    lines = [
        f"{len(aptas)} safra(s) aptas: {format_group(aptas)}.",
        f"{len(ressalvas)} safra(s) aptas com ressalvas: {format_group(ressalvas)}.",
        f"{len(inaptas)} safra(s) inaptas: {format_group(inaptas)}.",
        "",
        "Resumo numerico por safra:",
    ]

    for row in summary_rows:
        spatial_std = row["desvio_padrao_espacial_mediano"]
        spatial_std_text = "-" if spatial_std is None or math.isnan(spatial_std) else f"{spatial_std:.3f}"
        lines.append(
            f"{row['safra']}: {row['classificacao']} | cobertura={row['cobertura_temporal']:.1f}% | "
            f"lacunas_3s={row['lacunas_3_semanas']} | periodos={row['semanas_observadas']}/"
            f"{row['semanas_totais']} | amplitude={row['amplitude']:.3f} | "
            f"desvio_padrao_espacial_mediano={spatial_std_text}"
        )

    final_pdf = os.path.join(OUTPUT_DIR, "resumo_final_safras.pdf")
    save_simple_pdf("Resumo final das safras", lines, final_pdf)


def main():
    validate_configuration()
    initialize_earth_engine()

    log("Iniciando processamento...")
    geometry = extract_geometry_from_kmz(INPUT_DIR)
    log(f"Talhao carregado com geometria {geometry.geom_type}.")

    ee_geometry = shapely_to_ee_geometry(geometry)
    geometry_bounds = geometry.bounds

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary_rows = []
    for year_start in range(HARVEST_START_YEAR, HARVEST_END_YEAR + 1):
        year_end = year_start + 1
        log(f"Processando safra {year_start}/{year_end}...")
        df = build_harvest_dataframe(ee_geometry, year_start)
        summary_rows.append(
            save_harvest_outputs(
                df,
                ee_geometry,
                geometry_bounds,
                year_start,
            )
        )
        log(f"Safra {year_start}/{year_end} concluida.")

    save_final_summary(summary_rows)
    log(f"Processamento concluido. Resultados em '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
