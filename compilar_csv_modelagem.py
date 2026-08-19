import os
from datetime import datetime

import pandas as pd

import curvas_dnvi_das as ndvi_module
import dados_climaticos_das as clima_module


INPUT_DIR = globals().get("INPUT_DIR", "inputs")
OUTPUT_DIR = globals().get("OUTPUT_CSV_DIR", "outputs_csv")
OUTPUT_FILENAME = globals().get("OUTPUT_CSV_FILENAME", "dados_modelagem_ndvi.csv")
GEE_PROJECT_ID = globals().get("GEE_PROJECT_ID", os.getenv("EE_PROJECT_ID", "ndvi-estudos"))

DEFAULT_SAFRAS_PADRONIZADAS = [
    {"safra": "2018-2019", "inicio": "2018-11-15", "fim": "2019-04-01"},
    {"safra": "2019-2020", "inicio": "2019-11-12", "fim": "2020-04-05"},
    {"safra": "2021-2022", "inicio": "2021-11-10", "fim": "2022-04-15"},
    {"safra": "2022-2023", "inicio": "2022-11-01", "fim": "2023-04-04"},
    {"safra": "2023-2024", "inicio": "2023-11-14", "fim": "2024-04-02"},
    {"safra": "2025-2026", "inicio": "2025-11-11", "fim": "2026-04-10"},
]

SAFRAS_PADRONIZADAS = globals().get("SAFRAS_PADRONIZADAS", DEFAULT_SAFRAS_PADRONIZADAS)


def log(message: str):
    print(f"[CSV-MODELAGEM] {message}", flush=True)


def validate_configuration():
    if not SAFRAS_PADRONIZADAS:
        raise ValueError("Adicione ao menos uma safra em SAFRAS_PADRONIZADAS.")

    seen = set()
    for safra in SAFRAS_PADRONIZADAS:
        for key in ("safra", "inicio", "fim"):
            if key not in safra:
                raise ValueError(f"Cada safra deve conter '{key}'.")

        inicio = datetime.strptime(safra["inicio"], "%Y-%m-%d").date()
        fim = datetime.strptime(safra["fim"], "%Y-%m-%d").date()
        if fim < inicio:
            raise ValueError(f"Data final anterior ao inicio na safra {safra['safra']}.")
        if safra["safra"] in seen:
            raise ValueError(f"Safra duplicada em SAFRAS_PADRONIZADAS: {safra['safra']}.")
        seen.add(safra["safra"])


def percent(value: int, total: int):
    if total == 0:
        return 100.0
    return (value / total) * 100


def build_ndvi_frame(ee_geometry):
    all_rows = []
    total_periods = sum(
        len(
            ndvi_module.build_das_periods(
                datetime.strptime(safra["inicio"], "%Y-%m-%d").date(),
                datetime.strptime(safra["fim"], "%Y-%m-%d").date(),
            )
        )
        for safra in SAFRAS_PADRONIZADAS
    )

    processed_periods = 0
    for safra_config in SAFRAS_PADRONIZADAS:
        safra = safra_config["safra"]
        inicio = datetime.strptime(safra_config["inicio"], "%Y-%m-%d").date()
        fim = datetime.strptime(safra_config["fim"], "%Y-%m-%d").date()
        periods = ndvi_module.build_das_periods(inicio, fim)

        for period in periods:
            stats = ndvi_module.resolve_period_stats(
                ee_geometry,
                period["inicio"],
                period["fim"],
                inicio,
                fim,
            )

            processed_periods += 1
            progress_text = f"{percent(processed_periods, total_periods):.1f}%"
            ndvi_text = "NaN" if stats["status"] != "ok" else f"{stats['ndvi_median']:.4f}"
            log(f"NDVI {safra} | DAS_{period['das']:03d} | progresso {progress_text} | valor={ndvi_text}")

            all_rows.append(
                {
                    "safra": safra,
                    "DAS_atual": period["das"],
                    "NDVI t": stats["ndvi_median"] if stats["status"] == "ok" else pd.NA,
                }
            )

    ndvi_df = pd.DataFrame(all_rows).sort_values(["safra", "DAS_atual"]).reset_index(drop=True)
    ndvi_df["NDVI t-1"] = ndvi_df.groupby("safra")["NDVI t"].shift(1)
    return ndvi_df


def build_climate_frame(latitude: float, longitude: float):
    overall_start = min(
        datetime.strptime(safra["inicio"], "%Y-%m-%d").date()
        for safra in SAFRAS_PADRONIZADAS
    )
    overall_end = max(
        datetime.strptime(safra["fim"], "%Y-%m-%d").date()
        for safra in SAFRAS_PADRONIZADAS
    )

    daily_df = clima_module.fetch_nasa_power_daily(latitude, longitude, overall_start, overall_end)

    rows = []
    for safra_config in SAFRAS_PADRONIZADAS:
        safra = safra_config["safra"]
        inicio = datetime.strptime(safra_config["inicio"], "%Y-%m-%d").date()
        fim = datetime.strptime(safra_config["fim"], "%Y-%m-%d").date()
        periods = clima_module.build_das_periods(inicio, fim)

        for period in periods:
            row = clima_module.aggregate_period_row(safra, period, daily_df)
            rows.append(
                {
                    "safra": row["Safra"],
                    "DAS_atual": row["DAS"],
                    "precipitacao_ac": row["Precipitacao_Acumulada"],
                    "temperatura": row["Temperatura_Media"],
                    "radiacao": row["Radiacao_Acumulada"],
                    "umidade": row["Umidade_Relativa_Media"],
                    "vento ": row["Vento_Medio"],
                }
            )

    return pd.DataFrame(rows).sort_values(["safra", "DAS_atual"]).reset_index(drop=True)


def build_final_dataset(ndvi_df: pd.DataFrame, climate_df: pd.DataFrame):
    final_df = ndvi_df.merge(climate_df, on=["safra", "DAS_atual"], how="outer")
    final_df = final_df[
        [
            "safra",
            "DAS_atual",
            "NDVI t-1",
            "NDVI t",
            "precipitacao_ac",
            "temperatura",
            "radiacao",
            "umidade",
            "vento ",
        ]
    ]
    final_df = final_df.sort_values(["safra", "DAS_atual"]).reset_index(drop=True)
    return final_df


def main():
    validate_configuration()

    ndvi_module.INPUT_DIR = INPUT_DIR
    ndvi_module.GEE_PROJECT_ID = GEE_PROJECT_ID
    ndvi_module.SAFRAS_PADRONIZADAS = SAFRAS_PADRONIZADAS
    ndvi_module.validate_configuration()
    ndvi_module.initialize_earth_engine()

    geometry = ndvi_module.extract_geometry_from_kmz(INPUT_DIR)
    ee_geometry = ndvi_module.shapely_to_ee_geometry(geometry)
    ndvi_df = build_ndvi_frame(ee_geometry)

    clima_module.INPUT_DIR = INPUT_DIR
    clima_module.SAFRAS_PADRONIZADAS = SAFRAS_PADRONIZADAS
    clima_module.validate_configuration()
    latitude, longitude = clima_module.extract_centroid_from_kmz(INPUT_DIR)
    climate_df = build_climate_frame(latitude, longitude)

    final_df = build_final_dataset(ndvi_df, climate_df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    final_df.to_csv(output_path, index=False, na_rep="NaN")

    log(
        f"100.0% concluido. CSV consolidado salvo em '{output_path}' "
        f"com {len(final_df)} linhas."
    )


if __name__ == "__main__":
    main()
