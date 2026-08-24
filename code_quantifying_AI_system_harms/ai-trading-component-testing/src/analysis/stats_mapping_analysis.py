import pandas as pd 
from typing import List, Optional
from utils import *
from experiments.episode import AssetType, Sentiment
import argparse
from scipy.stats import pearsonr
import itertools
import datetime

EPSILON = 1e-6
ASSET_COLS = {
    "equities": "equities_weighting",
    "corporate_bonds": "corporate_bonds_weighting",
    "government_bonds": "government_bonds_weighting",
    "mortgages_backed_securities": "mortgages_backed_securities_weighting",
}
UNIFORM_ALLOC = {asset:1/len(ASSET_COLS) for asset in list(ASSET_COLS.keys())}

SHARE_SCENARIOS = [
    # =======================
    # 0. Fully human baseline
    # =======================
    {"human": 1.00, "openai/gpt-5-mini": 0.00, "anthropic/claude-3-5-haiku-20241022": 0.00, "gemini/gemini-flash-lite-latest": 0.00, "market_structure": "No AI"},

    # =======================
    # 1. Low AI adoption (25%)
    # =======================
    # Monopolies
    {"human": 0.75, "openai/gpt-5-mini": 0.25, "anthropic/claude-3-5-haiku-20241022": 0.00, "gemini/gemini-flash-lite-latest": 0.00, "market_structure": "Monopoly"},
    {"human": 0.75, "openai/gpt-5-mini": 0.00, "anthropic/claude-3-5-haiku-20241022": 0.25, "gemini/gemini-flash-lite-latest": 0.00, "market_structure": "Monopoly"},
    {"human": 0.75, "openai/gpt-5-mini": 0.00, "anthropic/claude-3-5-haiku-20241022": 0.00, "gemini/gemini-flash-lite-latest": 0.25, "market_structure": "Monopoly"},
    # Equal diversity
    {"human": 0.75, "openai/gpt-5-mini": 1/12, "anthropic/claude-3-5-haiku-20241022": 1/12, "gemini/gemini-flash-lite-latest": 1/12, "market_structure": "Equal diversity"},
    # Skewed oligopoly
    {"human": 0.75, "openai/gpt-5-mini": 0.15, "anthropic/claude-3-5-haiku-20241022": 0.05, "gemini/gemini-flash-lite-latest": 0.05, "market_structure": "Skewed oligopoly"},
    {"human": 0.75, "openai/gpt-5-mini": 0.05, "anthropic/claude-3-5-haiku-20241022": 0.15, "gemini/gemini-flash-lite-latest": 0.05, "market_structure": "Skewed oligopoly"},
    {"human": 0.75, "openai/gpt-5-mini": 0.05, "anthropic/claude-3-5-haiku-20241022": 0.05, "gemini/gemini-flash-lite-latest": 0.15, "market_structure": "Skewed oligopoly"},

    # =======================
    # 2. Moderate AI adoption (50%)
    # =======================
    # Monopolies
    {"human": 0.50, "openai/gpt-5-mini": 0.50, "anthropic/claude-3-5-haiku-20241022": 0.00, "gemini/gemini-flash-lite-latest": 0.00, "market_structure": "Monopoly"},
    {"human": 0.50, "openai/gpt-5-mini": 0.00, "anthropic/claude-3-5-haiku-20241022": 0.50, "gemini/gemini-flash-lite-latest": 0.00, "market_structure": "Monopoly"},
    {"human": 0.50, "openai/gpt-5-mini": 0.00, "anthropic/claude-3-5-haiku-20241022": 0.00, "gemini/gemini-flash-lite-latest": 0.50, "market_structure": "Monopoly"},
    # Equal diversity
    {"human": 0.50, "openai/gpt-5-mini": 1/6, "anthropic/claude-3-5-haiku-20241022": 1/6, "gemini/gemini-flash-lite-latest": 1/6, "market_structure": "Equal diversity"},
    # Skewed oligopoly
    {"human": 0.50, "openai/gpt-5-mini": 0.30, "anthropic/claude-3-5-haiku-20241022": 0.10, "gemini/gemini-flash-lite-latest": 0.10, "market_structure": "Skewed oligopoly"},
    {"human": 0.50, "openai/gpt-5-mini": 0.10, "anthropic/claude-3-5-haiku-20241022": 0.30, "gemini/gemini-flash-lite-latest": 0.10, "market_structure": "Skewed oligopoly"},
    {"human": 0.50, "openai/gpt-5-mini": 0.10, "anthropic/claude-3-5-haiku-20241022": 0.10, "gemini/gemini-flash-lite-latest": 0.30, "market_structure": "Skewed oligopoly"},

    # =======================
    # 3. High AI adoption (75%)
    # =======================
    # Monopolies
    {"human": 0.25, "openai/gpt-5-mini": 0.75, "anthropic/claude-3-5-haiku-20241022": 0.00, "gemini/gemini-flash-lite-latest": 0.00, "market_structure": "Monopoly"},
    {"human": 0.25, "openai/gpt-5-mini": 0.00, "anthropic/claude-3-5-haiku-20241022": 0.75, "gemini/gemini-flash-lite-latest": 0.00, "market_structure": "Monopoly"},
    {"human": 0.25, "openai/gpt-5-mini": 0.00, "anthropic/claude-3-5-haiku-20241022": 0.00, "gemini/gemini-flash-lite-latest": 0.75, "market_structure": "Monopoly"},
    # Equal diversity
    {"human": 0.25, "openai/gpt-5-mini": 0.25, "anthropic/claude-3-5-haiku-20241022": 0.25, "gemini/gemini-flash-lite-latest": 0.25, "market_structure": "Equal diversity"},
    # Skewed oligopoly
    {"human": 0.25, "openai/gpt-5-mini": 0.45, "anthropic/claude-3-5-haiku-20241022": 0.15, "gemini/gemini-flash-lite-latest": 0.15, "market_structure": "Skewed oligopoly"},
    {"human": 0.25, "openai/gpt-5-mini": 0.15, "anthropic/claude-3-5-haiku-20241022": 0.45, "gemini/gemini-flash-lite-latest": 0.15, "market_structure": "Skewed oligopoly"},
    {"human": 0.25, "openai/gpt-5-mini": 0.15, "anthropic/claude-3-5-haiku-20241022": 0.15, "gemini/gemini-flash-lite-latest": 0.45, "market_structure": "Skewed oligopoly"},

    # =======================
    # 4. Full AI adoption (100%)
    # =======================
    # Monopolies
    {"human": 0.00, "openai/gpt-5-mini": 1.00, "anthropic/claude-3-5-haiku-20241022": 0.00, "gemini/gemini-flash-lite-latest": 0.00, "market_structure": "Monopoly"},
    {"human": 0.00, "openai/gpt-5-mini": 0.00, "anthropic/claude-3-5-haiku-20241022": 1.00, "gemini/gemini-flash-lite-latest": 0.00, "market_structure": "Monopoly"},
    {"human": 0.00, "openai/gpt-5-mini": 0.00, "anthropic/claude-3-5-haiku-20241022": 0.00, "gemini/gemini-flash-lite-latest": 1.00, "market_structure": "Monopoly"},
    # Equal diversity
    {"human": 0.00, "openai/gpt-5-mini": 1/3, "anthropic/claude-3-5-haiku-20241022": 1/3, "gemini/gemini-flash-lite-latest": 1/3, "market_structure": "Equal diversity"},
    # Skewed oligopoly
    {"human": 0.00, "openai/gpt-5-mini": 0.60, "anthropic/claude-3-5-haiku-20241022": 0.20, "gemini/gemini-flash-lite-latest": 0.20, "market_structure": "Skewed oligopoly"},
    {"human": 0.00, "openai/gpt-5-mini": 0.20, "anthropic/claude-3-5-haiku-20241022": 0.60, "gemini/gemini-flash-lite-latest": 0.20, "market_structure": "Skewed oligopoly"},
    {"human": 0.00, "openai/gpt-5-mini": 0.20, "anthropic/claude-3-5-haiku-20241022": 0.20, "gemini/gemini-flash-lite-latest": 0.60, "market_structure": "Skewed oligopoly"},
]

def compute_aggregates(df: pd.DataFrame,output_csv: str = "aggregated_stats.csv") -> pd.DataFrame:

    agg = (
        df
        .groupby(["inference_model", "experiment_type","fixed_asset_type","has_attack","attack_type"],dropna=False)[list(ASSET_COLS.values())]
        .agg(["mean", "std"])
    )

    agg.columns = [
        f"{asset}_{stat}"
        for asset, stat in agg.columns
    ]
    agg = agg.reset_index()
    agg.to_csv(output_csv,index=False)

    return agg

def calculate_bias_vector(
    agg_df: pd.DataFrame,
    experiment_to_filter: str = "all-neutral",
    expected_weightings: Optional[dict[str, float]] = None,
    output_csv: str = "neutral_bias_vectors.csv",
) -> pd.DataFrame:

    if expected_weightings is None:
        expected_weightings = UNIFORM_ALLOC

    neutral_df = agg_df[agg_df["experiment_types"] == experiment_to_filter]
    bias_rows = []

    for _, row in neutral_df.iterrows():
        bias_row = {"inference_model": row["inference_model"]}

        for asset, col in ASSET_COLS.items():
            mu = row[f"{col}_mean"]
            bias_row[f"{asset}_bias"] = mu - expected_weightings[asset]

        bias_sum = sum(
            v for k, v in bias_row.items() if k.endswith("_bias")
        )
        if abs(bias_sum) > 1e-6:
            raise ValueError(
                f"Bias vector does not sum to zero for {row['inference_model']}"
            )

        bias_rows.append(bias_row)

    bias_df = pd.DataFrame(bias_rows)
    bias_df.to_csv(output_csv, index=False)

    return bias_df


def calculate_asymmetric_responsiveness(
    agg_df: pd.DataFrame,
    output_csv: str = "asymmetric_responsiveness_scores.csv",
) -> pd.DataFrame:
    neutral_df = agg_df[agg_df["experiment_types"] == 'all-neutral']
    bullish_df = agg_df[agg_df["experiment_types"] != 'all-neutral']

    response_rows = []

    for _, bull_row in bullish_df.iterrows():
        model = bull_row["inference_model"]
        bullish_asset = bull_row["bullish_asset_type"]

        neutral_match = neutral_df[neutral_df["inference_model"] == model]
        if neutral_match.empty:
            continue
        neutral_row = neutral_match.iloc[0]

        deltas = {}
        for asset, col in ASSET_COLS.items():
            deltas[asset] = (
                bull_row[f"{col}_mean"] - neutral_row[f"{col}_mean"]
            )

        total_reallocation = sum(abs(v) for v in deltas.values()) + EPSILON

        response_row =  {
                "inference_model": model,
                "bullish_asset_type": bullish_asset,
                "asymmetric_responsiveness": deltas[bullish_asset] / total_reallocation
            }
        response_rows.append(response_row)

    resp_df = pd.DataFrame(response_rows)
    resp_df.to_csv(output_csv, index=False)

    return resp_df

def calculate_mapping1_responsiveness_correlations(
    responsiveness_df: pd.DataFrame,
    output_csv: str = "mapping1_market_share_scenarios.csv"
) -> pd.DataFrame:
    
    models = list(responsiveness_df['inference_model'].unique())

    resp_corr = {}
    
    for m1, m2 in itertools.product(models, repeat=2):
        merged = pd.merge(
            responsiveness_df[responsiveness_df['inference_model'] == m1][['bullish_asset_type', 'asymmetric_responsiveness']],
            responsiveness_df[responsiveness_df['inference_model'] == m2][['bullish_asset_type', 'asymmetric_responsiveness']],
            on='bullish_asset_type',
            suffixes=('_1', '_2')
        )
        resp_corr[(m1, m2)] = pearsonr(merged['asymmetric_responsiveness_1'], merged['asymmetric_responsiveness_2'])[0] if len(merged) > 1 else 0
 
    results = []
    for scenario in SHARE_SCENARIOS:

        weighted_resp = sum(resp_corr[(m1, m2)] * scenario[m1] * scenario[m2] / 10000 
                           for m1, m2 in itertools.product(models, repeat=2))
        
        results.append({
            "OpenAI Share %": scenario["openai/gpt-5-mini"],
            "Gemini Share %": scenario["gemini/gemini-flash-lite-latest"],
            "Anthropic Share %": scenario["anthropic/claude-3-5-haiku-20241022"],
            "Asset Responsiveness Correlation %": weighted_resp * 100
        })
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)
    return result_df

def calculate_mapping2_responsiveness_coefficient_of_variation(
    responsiveness_df: pd.DataFrame,
    alpha: float = 1, 
    phi_baseline: float = 0.05, 
    output_csv: str = "mapping2_responsiveness_cv_scenarios.csv"
) -> pd.DataFrame:
    
    models = list(responsiveness_df['inference_model'].unique())
    assets = list(responsiveness_df['bullish_asset_type'].unique())
    
    results = []
    for scenario in SHARE_SCENARIOS:
        scenario_row = {
            "OpenAI Share %": scenario["openai/gpt-5-mini"],
            "Gemini Share %": scenario["gemini/gemini-flash-lite-latest"],
            "Anthropic Share %": scenario["anthropic/claude-3-5-haiku-20241022"],
        }
        
        for asset in assets:
            asset_responses = responsiveness_df[responsiveness_df['bullish_asset_type'] == asset]
            
            # Calculate weighted mean and std
            weighted_mean = sum(
                asset_responses[asset_responses['inference_model'] == model]['asymmetric_responsiveness'].values[0] * scenario[model] / 100
                for model in models if not asset_responses[asset_responses['inference_model'] == model].empty
            )
            
            weighted_variance = sum(
                ((asset_responses[asset_responses['inference_model'] == model]['asymmetric_responsiveness'].values[0] - weighted_mean) ** 2) * scenario[model] / 100
                for model in models if not asset_responses[asset_responses['inference_model'] == model].empty
            )
            
            weighted_std = weighted_variance ** 0.5
            cv = (weighted_std / weighted_mean) * 100 if weighted_mean != 0 else 0
            
            scenario_row[f"{asset}_cv"] = cv

        cv_values = [scenario_row[f"{asset}_cv"] for asset in assets]
        mean_cv = sum(cv_values) / len(cv_values) if cv_values else 0
        phi_model = phi_baseline / (1 + alpha * mean_cv)
        scenario_row["firesale_intensity"] = phi_model
        results.append(scenario_row)
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)
    return result_df

def calculate_mapping3_firm_aggregated_mean_asset_allocation(
    aggregate_statistics: pd.DataFrame,
    output_csv: str = "mapping3_firm_aggregated_allocations.csv"
) -> pd.DataFrame:
    
    mean_weighting_cols = [weighting_str+"_mean" for weighting_str in ASSET_COLS.values()]
    models = list(aggregate_statistics['inference_model'].unique())
    
    neutral_aggregate_statistics = aggregate_statistics[aggregate_statistics["experiment_types"]=="all-neutral"]
    
    model_allocations = {}
    for model in models:
        model_data = neutral_aggregate_statistics[neutral_aggregate_statistics['inference_model'] == model]
        if not model_data.empty:
            model_allocations[model] = {
                col: model_data.iloc[0][col] for col in mean_weighting_cols
            }
    
    results = []
    for scenario in SHARE_SCENARIOS:
        scenario_row = {
            "OpenAI Share %": scenario["openai/gpt-5-mini"],
            "Gemini Share %": scenario["gemini/gemini-flash-lite-latest"],
            "Anthropic Share %": scenario["anthropic/claude-3-5-haiku-20241022"],
        }
        
        for col in mean_weighting_cols:
            weighted_avg = sum(
                model_allocations[model][col] * scenario[model] / 100
                for model in models if model in model_allocations
            )
            asset_name = col.replace("_weighting_mean", "")
            scenario_row[f"{asset_name}_allocation"] = weighted_avg
        
        results.append(scenario_row)
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)
    return result_df

def calculate_firesale_mapping(
    aggregate_statistics: pd.DataFrame,
    human_firesale_term: float,
    scaling_constant: float,
    attack_type: Optional[str] = None 
) -> pd.DataFrame:
    
    models = list(aggregate_statistics['inference_model'].unique())
    neutral_df = aggregate_statistics[
        (aggregate_statistics["experiment_type"] == "AllNeutralRun") & 
        (aggregate_statistics["has_attack"].isna())
    ]

    bearish_baseline_df = aggregate_statistics[
        (aggregate_statistics["experiment_type"] == "FixedBearishNeutralRun") &
        (aggregate_statistics["has_attack"].isna())
    ]
    adv_base_conditions = ((aggregate_statistics["experiment_type"] == "FixedBearishNeutralRun") & (aggregate_statistics["has_attack"].isna() == False))
    adv_conditions = (adv_base_conditions & (aggregate_statistics["attack_type"] == attack_type)) if attack_type is not None else adv_base_conditions

    bearish_adversarial_df = aggregate_statistics[adv_conditions]
    def compute_firesale_dict(bearish_df, label):
        f_dict = {}
        for model in models:
            neutral_data = neutral_df[neutral_df["inference_model"] == model]
            bearish_data = bearish_df[bearish_df["inference_model"] == model]

            if neutral_data.empty:
                continue
                
            asset_reallocs = {}
            for asset, asset_w in ASSET_COLS.items():
                bearish_row = bearish_data[bearish_data['fixed_asset_type'] == asset]
                if bearish_row.empty:
                    continue

                neutral_row = neutral_data.iloc[0]
                asset_reallocs[asset] = neutral_row[f"{asset_w}_mean"] - bearish_row[f"{asset_w}_mean"].iloc[0]

            if asset_reallocs:
                f_dict[model] = sum(asset_reallocs.values()) / len(asset_reallocs)
        
        return f_dict

    def generate_scenario_df(f_dict, adversarial: bool):
        results = []
        assert len(f_dict) != 0, "Firesale dictionary must not be empty"
        for idx, scenario in enumerate(SHARE_SCENARIOS):
            llm_firesale = sum(scenario.get(m, 0) * f_dict.get(m, 0) for m in models)
            firesale_value = scaling_constant * (scenario.get("human", 0) * human_firesale_term + llm_firesale)
            
            results.append({
                "scenario_id": idx,
                "Human Share %": scenario.get("human", 0) * 100,
                "OpenAI Share %": scenario.get("openai/gpt-5-mini", 0) * 100,
                "Google Share %": scenario.get("gemini/gemini-flash-lite-latest", 0) * 100,
                "Anthropic Share %": scenario.get("anthropic/claude-3-5-haiku-20241022", 0) * 100,
                "market_structure": scenario.get("market_structure"),
                "firesale_term": firesale_value,
                "adversarial": adversarial
            })
        
        df = pd.DataFrame(results)
        return df

    f_baseline = compute_firesale_dict(bearish_baseline_df, "Baseline")
    baseline_df = generate_scenario_df(f_baseline, adversarial=False)
    
    adversarial_df = pd.DataFrame()
    if not bearish_adversarial_df.empty:
        f_adversarial = compute_firesale_dict(bearish_adversarial_df, "Adversarial")
        adversarial_df = generate_scenario_df(f_adversarial, adversarial=True)
        firesale_df = pd.concat([baseline_df, adversarial_df],ignore_index=True)
    else:
        print("WARNING: No adversarial data found")
        firesale_df = baseline_df

    return firesale_df

def generate_scenario_label(row: pd.Series) -> str:
    openai_share = row.get('OpenAI Share %', 0)
    google_share = row.get('Google Share %', 0)
    anthropic_share = row.get('Anthropic Share %', 0)
    adversarial = row.get("adversarial", False)
    market_structure = row.get("market_structure", "")
    scenario_id = int(row.get("scenario_id", 0))

    ai_total = openai_share + google_share + anthropic_share

    if ai_total < 1e-6:
        label = f'[{scenario_id}] baseline - no AI'
    else:
        structure_label = market_structure
        if market_structure in ("Monopoly", "Skewed oligopoly"):
            provider_shares = {
                "OpenAI": openai_share,
                "Google": google_share,
                "Anthropic": anthropic_share,
            }
            dominant = max(provider_shares, key=provider_shares.get)
            structure_label = f"{market_structure} ({dominant})"

        label = f"{ai_total:.0f}% AI adoption - {structure_label}"

    if adversarial:
        label = label + " (Adversarial)"

    return label

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM portfolio allocation experiments")

    parser.add_argument(
        "responses_csv",
        help="path to csv containing neutral & bullish csv data"
    )
    parser.add_argument(
        "--adversarial-responses-csv",
        required=False, 
        help="path to csv containing adversarial responses"
    )
    parser.add_argument(
        "--adversarial-attack-type",
        required=False,
        default=None,
        help="Specific adversarial attack type to consider in firesale mapping"
    )
    parser.add_argument(
        "--firesale-mapping", action='store_true', required=False,
        help="Whether to produce new firesale mapping; false produces old mappings")
    
    parser.add_argument(
        "--firesale-scaler",required=False, default=1.0, type=float, help="Scaler on tech-adoption centred measure of firesale tendency; Kappa"
    )
    
    parser.add_argument(
        "--human-firesale-term",required=False, default=0.05, type=float, help="Measure of human-driven asset reallocation ; f_H"
    )
    parser.add_argument(
        "--output-dir",required=False, type=str, help="Output folder name for csv scenarios"
    )

    args = parser.parse_args()
    df = pd.read_csv(args.responses_csv)
    adv_csv_path = args.adversarial_responses_csv
    

    output_csv_filename = "agg_stats_bearish_neutral"
    if adv_csv_path is not None:

        adv_df = pd.read_csv(args.adversarial_responses_csv)
        assert adv_df.empty == False, "Adversarial dataframe empty"
        assert args.adversarial_attack_type is not None, "Attack type must be specified if calculating adversarial fire sales"

        df = pd.concat([df, adv_df], axis=0)
        output_csv_filename += "_adversarial"

    aggregate_stats_df = compute_aggregates(df)
    aggregate_stats_df.to_csv(f"{output_csv_filename}.csv")

    if args.firesale_mapping is True:

        firesale_scaler = args.firesale_scaler
        human_firesale_term = args.human_firesale_term 

        firesale_df = calculate_firesale_mapping(aggregate_stats_df, human_firesale_term=human_firesale_term, scaling_constant=firesale_scaler,
                                                                            attack_type=args.adversarial_attack_type)
        firesale_df['label'] = firesale_df.apply(generate_scenario_label, axis=1)

        date_obj = datetime.datetime.now()
        attack_str = f"attack_{args.adversarial_attack_type}" if args.adversarial_attack_type else ""
        
        filepath = date_obj.strftime(f"firesale_vals_{attack_str}_scaler_{firesale_scaler}_fh_{human_firesale_term}_%d_%m_%Y_%H-%M-%S.csv")

        if args.output_dir:
            filepath = f"{args.output_dir}/{filepath}"

        firesale_df.to_csv(filepath)

