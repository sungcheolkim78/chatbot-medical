"""
Score plot for chatbot evaluation
"""

import pandas as pd
import seaborn as sns
import glob
from typing import List
from components import setup_logging
import logging

setup_logging()


def load_llm_score_data(
    pattern: str = "evaluation/chatbot_results/llm_score_*.csv",
) -> pd.DataFrame:
    """Load and concatenate all LLM score CSVs matching the given pattern."""
    logging.info(f"Loading CSV files with pattern: {pattern}")
    csv_files = glob.glob(pattern)
    if not csv_files:
        logging.error(f"No CSV files found for pattern: {pattern}")
        raise FileNotFoundError(f"No CSV files found for pattern: {pattern}")
    dfs = [pd.read_csv(file) for file in csv_files]
    logging.info(f"Loaded {len(dfs)} CSV files.")
    df = pd.concat(dfs, ignore_index=True)
    logging.info(f"Concatenated dataframe shape: {df.shape}")
    return df


def aggregate_scores(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """
    1. Average metrics over repeat_id for each (id, model_name).
    2. Return melted DataFrame with columns: id, model_name, Metric, Score.
    """
    logging.info(
        "Aggregating scores: averaging over repeat_id and then melting by metric."
    )
    df_avg_repeat = df.groupby(["id", "model_name"])[metrics].mean().reset_index()
    # df_avg_repeat.to_csv("evaluation/chatbot_results/metrics_mean_by_model_and_id.csv", index=False)
    # Melt the DataFrame
    df_melted = pd.melt(
        df_avg_repeat,
        id_vars=["id", "model_name"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Score",
    )
    # df_melted.to_csv("evaluation/chatbot_results/metrics_mean_by_model_and_id_melted.csv", index=False)
    df_melted = df_melted.sort_values("model_name")
    df_melted.to_csv("evaluation/chatbot_results/metrics_melted.csv", index=False)
    return df_melted


def plot_mean_with_stderr(
    results_df: pd.DataFrame, metrics: List[str], output_path: str
):
    """Plot mean with standard error bars for each metric and model, with correct alignment."""

    logging.info(f"Plotting mean with standard error. Output path: {output_path}")
    g = sns.catplot(
        data=results_df,
        kind="bar",
        x="Metric",
        y="Score",
        hue="model_name",
        errorbar="se",
        height=8,
        aspect=1.5,
        capsize=0.4,
        err_kws={"color": ".2", "linewidth": 2.5},
    )
    metrics[2] = "User experience"
    g.set_axis_labels("", "Score")
    g.set_xticklabels(metrics)
    g.set_titles("Grouped by Model")
    g.despine(left=True)
    g.savefig(output_path, dpi=300)
    logging.info(f"Plot saved to {output_path}")


def main():
    metrics = ["correctness", "response_time", "style"]
    try:
        df = load_llm_score_data()
        results_df = aggregate_scores(df, metrics)
        plot_mean_with_stderr(
            results_df,
            metrics,
            "evaluation/chatbot_results/metrics_mean_stderr_by_model.png",
        )
        logging.info("Script completed successfully.")
    except Exception as e:
        logging.error(f"Script failed: {e}")
        raise


if __name__ == "__main__":
    main()
