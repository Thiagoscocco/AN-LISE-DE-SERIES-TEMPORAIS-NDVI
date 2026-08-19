import glob
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd


INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs_dados_climaticos"
NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
DAS_STEP_DAYS = 15

SAFRAS_PADRONIZADAS = [
    {"safra": "2018-2019", "inicio": "2018-11-15", "fim": "2019-04-01"},
    {"safra": "2019-2020", "inicio": "2019-11-12", "fim": "2020-04-05"},
    {"safra": "2021-2022", "inicio": "2021-11-10", "fim": "2022-04-15"},
    {"safra": "2022-2023", "inicio": "2022-11-01", "fim": "2023-04-04"},
    {"safra": "2023-2024", "inicio": "2023-11-14", "fim": "2024-04-02"},
    {"safra": "2025-2026", "inicio": "2025-11-11", "fim": "2026-04-10"},
]

CLIMATE_VARIABLES = [
    {
        "api_name": "PRECTOTCORR",
        "column": "Precipitacao_Acumulada",
        "label": "Precipitacao acumulada",
        "unit": "mm/periodo",
        "aggregation": "sum",
        "plot_kind_single": "bar",
        "color": "#2f7ed8",
        "output_slug": "precipitacao",
    },
    {
        "api_name": "ALLSKY_SFC_SW_DWN",
        "column": "Radiacao_Acumulada",
        "label": "Radiacao solar acumulada",
        "unit": "kWh/m2/dia somado",
        "aggregation": "sum",
        "plot_kind_single": "line",
        "color": "#f39c12",
        "output_slug": "radiacao",
    },
    {
        "api_name": "T2M",
        "column": "Temperatura_Media",
        "label": "Temperatura media",
        "unit": "C",
        "aggregation": "mean",
        "plot_kind_single": "line",
        "color": "#c0392b",
        "output_slug": "temperatura",
    },
    {
        "api_name": "RH2M",
        "column": "Umidade_Relativa_Media",
        "label": "Umidade relativa media",
        "unit": "%",
        "aggregation": "mean",
        "plot_kind_single": "line",
        "color": "#16a085",
        "output_slug": "umidade_relativa",
    },
    {
        "api_name": "WS2M",
        "column": "Vento_Medio",
        "label": "Velocidade do vento media",
        "unit": "m/s",
        "aggregation": "mean",
        "plot_kind_single": "line",
        "color": "#8e44ad",
        "output_slug": "vento",
    },
]

FILL_VALUES = {-999, -999.0, -99, -99.0}


def log(message: str):
    print(f"[CLIMA-DAS] {message}", flush=True)


def validate_configuration():
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


def extract_kml_text_from_kmz(input_dir: str):
    kmz_files = glob.glob(os.path.join(input_dir, "*.kmz"))
    if not kmz_files:
        raise FileNotFoundError("Nenhum arquivo .kmz encontrado na pasta inputs.")

    kmz_path = kmz_files[0]
    with zipfile.ZipFile(kmz_path, "r") as zip_ref:
        kml_names = [name for name in zip_ref.namelist() if name.lower().endswith(".kml")]
        if not kml_names:
            raise FileNotFoundError("Nenhum arquivo .kml encontrado dentro do KMZ.")
        with zip_ref.open(kml_names[0]) as kml_file:
            return kml_file.read().decode("utf-8")


def parse_coordinates_text(coordinates_text: str):
    coordinates = []
    for point_text in coordinates_text.strip().split():
        parts = point_text.split(",")
        if len(parts) < 2:
            continue
        longitude = float(parts[0])
        latitude = float(parts[1])
        coordinates.append((longitude, latitude))
    return coordinates


def polygon_centroid(coordinates):
    if len(coordinates) < 3:
        raise ValueError("Poligono invalido no KML: menos de 3 vertices.")

    if coordinates[0] != coordinates[-1]:
        coordinates = [*coordinates, coordinates[0]]

    double_area = 0.0
    centroid_x = 0.0
    centroid_y = 0.0

    for index in range(len(coordinates) - 1):
        x0, y0 = coordinates[index]
        x1, y1 = coordinates[index + 1]
        cross = (x0 * y1) - (x1 * y0)
        double_area += cross
        centroid_x += (x0 + x1) * cross
        centroid_y += (y0 + y1) * cross

    if double_area == 0:
        longitudes = [point[0] for point in coordinates[:-1]]
        latitudes = [point[1] for point in coordinates[:-1]]
        return sum(longitudes) / len(longitudes), sum(latitudes) / len(latitudes)

    area_factor = 3 * double_area
    return centroid_x / area_factor, centroid_y / area_factor


def extract_centroid_from_kmz(input_dir: str):
    kml_text = extract_kml_text_from_kmz(input_dir)
    root = ET.fromstring(kml_text)
    namespace = {"kml": "http://www.opengis.net/kml/2.2"}

    coordinate_nodes = root.findall(".//kml:Polygon//kml:outerBoundaryIs//kml:LinearRing//kml:coordinates", namespace)
    if not coordinate_nodes:
        raise ValueError("Nenhum poligono com coordenadas foi encontrado no KML.")

    centroids = []
    for node in coordinate_nodes:
        if not node.text or not node.text.strip():
            continue
        coordinates = parse_coordinates_text(node.text)
        centroids.append(polygon_centroid(coordinates))

    if not centroids:
        raise ValueError("Nenhuma coordenada valida foi encontrada no KML.")

    avg_longitude = sum(point[0] for point in centroids) / len(centroids)
    avg_latitude = sum(point[1] for point in centroids) / len(centroids)
    return avg_latitude, avg_longitude


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


def build_power_url(latitude: float, longitude: float, start_date: date, end_date: date):
    params = {
        "parameters": ",".join(variable["api_name"] for variable in CLIMATE_VARIABLES),
        "community": "AG",
        "longitude": f"{longitude:.6f}",
        "latitude": f"{latitude:.6f}",
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON",
    }
    return f"{NASA_POWER_BASE_URL}?{urllib.parse.urlencode(params)}"


def parse_power_parameter(parameter_payload):
    if not isinstance(parameter_payload, dict):
        return {}

    normalized = {}
    for key, value in parameter_payload.items():
        parsed_date = datetime.strptime(str(key), "%Y%m%d").date()
        if value in FILL_VALUES:
            normalized[parsed_date] = None
        else:
            normalized[parsed_date] = value
    return normalized


def fetch_nasa_power_daily(latitude: float, longitude: float, start_date: date, end_date: date):
    url = build_power_url(latitude, longitude, start_date, end_date)
    log(
        "Consultando NASA POWER "
        f"de {start_date.isoformat()} ate {end_date.isoformat()} "
        f"para lat={latitude:.6f}, lon={longitude:.6f}"
    )

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Falha ao consultar NASA POWER: {exc}") from exc

    parameters = payload.get("properties", {}).get("parameter", {})
    if not parameters:
        raise RuntimeError("Resposta da NASA POWER sem dados em 'properties.parameter'.")

    data = {}
    for variable in CLIMATE_VARIABLES:
        parameter_payload = parameters.get(variable["api_name"], {})
        data[variable["column"]] = parse_power_parameter(parameter_payload)

    daily_df = pd.DataFrame(data)
    daily_df.index.name = "Data"
    daily_df = daily_df.sort_index()
    daily_df.index = pd.to_datetime(daily_df.index)
    return daily_df


def aggregate_series(series: pd.Series, aggregation: str):
    if aggregation == "sum":
        return series.sum(min_count=1)
    if aggregation == "mean":
        return series.mean()
    raise ValueError(f"Agregacao nao suportada: {aggregation}")


def aggregate_period_row(safra: str, period, daily_df: pd.DataFrame):
    window_df = daily_df.loc[
        pd.Timestamp(period["inicio"]):pd.Timestamp(period["fim"]),
        :,
    ]

    row = {
        "Safra": safra,
        "DAS": period["das"],
        "Rotulo_DAS": f"DAS_{period['das']:03d}",
        "Data_Inicial_Base": period["inicio"].isoformat(),
        "Data_Final_Base": period["fim"].isoformat(),
        "Dias_No_Periodo": int((period["fim"] - period["inicio"]).days + 1),
    }

    for variable in CLIMATE_VARIABLES:
        row[variable["column"]] = aggregate_series(
            window_df[variable["column"]],
            variable["aggregation"],
        )

    return row


def plot_single_variable(df: pd.DataFrame, safra: str, variable, output_path: str):
    plt.figure(figsize=(12, 5))
    x_values = df["DAS"]
    y_values = df[variable["column"]]

    if variable["plot_kind_single"] == "bar":
        plt.bar(x_values, y_values, width=10, color=variable["color"], alpha=0.85)
    else:
        plt.plot(
            x_values,
            y_values,
            color=variable["color"],
            linewidth=2,
            marker="o",
            markersize=5,
        )

    plt.title(f"{variable['label']} por DAS - Safra {safra}")
    plt.xlabel("DAS")
    plt.ylabel(f"{variable['label']} ({variable['unit']})")
    plt.xticks(x_values)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_overlay(all_data: pd.DataFrame, variable, output_path: str):
    plt.figure(figsize=(13, 6))
    for safra, df_safra in all_data.groupby("Safra"):
        plt.plot(
            df_safra["DAS"],
            df_safra[variable["column"]],
            linewidth=2,
            marker="o",
            markersize=4,
            label=safra,
        )

    plt.title(f"{variable['label']} por DAS - Safras sobrepostas")
    plt.xlabel("DAS")
    plt.ylabel(f"{variable['label']} ({variable['unit']})")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_single_harvest(safra_config, daily_df: pd.DataFrame):
    safra = safra_config["safra"]
    inicio = datetime.strptime(safra_config["inicio"], "%Y-%m-%d").date()
    fim = datetime.strptime(safra_config["fim"], "%Y-%m-%d").date()
    periods = build_das_periods(inicio, fim)

    harvest_dir = os.path.join(OUTPUT_DIR, safra)
    os.makedirs(harvest_dir, exist_ok=True)

    rows = []
    for period in periods:
        row = aggregate_period_row(safra, period, daily_df)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["Data_Inicial_Base"] = pd.to_datetime(df["Data_Inicial_Base"])
    df["Data_Final_Base"] = pd.to_datetime(df["Data_Final_Base"])

    csv_path = os.path.join(harvest_dir, f"dados_climaticos_{safra.replace('-', '_')}.csv")
    df.to_csv(csv_path, index=False)

    for variable in CLIMATE_VARIABLES:
        graph_path = os.path.join(
            harvest_dir,
            f"{variable['output_slug']}_das_{safra.replace('-', '_')}.png",
        )
        plot_single_variable(df, safra, variable, graph_path)

    log(f"{safra} concluida com {len(df)} periodos DAS.")
    return df


def main():
    validate_configuration()

    latitude, longitude = extract_centroid_from_kmz(INPUT_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    overall_start = min(
        datetime.strptime(safra["inicio"], "%Y-%m-%d").date()
        for safra in SAFRAS_PADRONIZADAS
    )
    overall_end = max(
        datetime.strptime(safra["fim"], "%Y-%m-%d").date()
        for safra in SAFRAS_PADRONIZADAS
    )

    daily_df = fetch_nasa_power_daily(latitude, longitude, overall_start, overall_end)

    all_frames = []
    for safra_config in SAFRAS_PADRONIZADAS:
        all_frames.append(run_single_harvest(safra_config, daily_df))

    consolidated = pd.concat(all_frames, ignore_index=True)
    consolidated_path = os.path.join(OUTPUT_DIR, "dados_climaticos_das_consolidado.csv")
    consolidated.to_csv(consolidated_path, index=False)

    for variable in CLIMATE_VARIABLES:
        overlay_path = os.path.join(
            OUTPUT_DIR,
            f"{variable['output_slug']}_das_safras_sobrepostas.png",
        )
        plot_overlay(consolidated, variable, overlay_path)

    log(
        "100.0% concluido. Resultados salvos em "
        f"'{OUTPUT_DIR}' para lat={latitude:.6f}, lon={longitude:.6f}."
    )


if __name__ == "__main__":
    main()
