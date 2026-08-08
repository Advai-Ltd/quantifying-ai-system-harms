import os
import json
from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import numpy as np

# Asset column mappings
ASSET_WEIGHT_COLUMNS = {
    'Equities': 'equities_weighting',
    'Corporate Bonds': 'corporate_bonds_weighting',
    'Government Bonds': 'government_bonds_weighting',
    'MBS': 'mortgages_backed_securities_weighting'
}

class WeightingDistributionPlotter:
    """Plot distributions of asset weightings from experimental results."""
    
    def __init__(self, results_df: pd.DataFrame,
                 model_inf_filter: Optional[str] = None,
                 model_news_gen_filter: Optional[str] = None):
        self.df = results_df.copy()
        self._validate_columns()
        self._parse_article_fields()
        self.model_inf_filter = model_inf_filter
        self.model_news_gen_filter = model_news_gen_filter

        if self.model_inf_filter is not None:
            self._filter_by_inference_model(self.model_inf_filter)
        
        if self.model_news_gen_filter is not None:
            self._filter_by_article_models(self.model_news_gen_filter)
    
    def _validate_columns(self):
        """Check that required columns exist."""
        missing = [col for col in ASSET_WEIGHT_COLUMNS.values() if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def _parse_article_fields(self):
        for i in range(1, 5):
            field_col = f'article_{i}_fields'
            if field_col in self.df.columns:
                parsed = self.df[field_col].apply(lambda x: json.loads(x) if pd.notna(x) else {})
                self.df[f'article_{i}_sentiment'] = parsed.apply(lambda x: x.get('sentiment', None))
                self.df[f'article_{i}_model'] = parsed.apply(lambda x: x.get('model', None))
                self.df[f'article_{i}_asset_type'] = parsed.apply(lambda x: x.get('asset_type', None))
            
    def _filter_by_inference_model(self, model: str):
        if 'inference_model' not in self.df.columns:
            raise ValueError("No 'inference_model' column found")
        
        original_len = len(self.df)
        self.df = self.df[self.df['inference_model'] == model].copy()
        print(f"Filtered by inference model '{model}': {original_len} -> {len(self.df)} rows")
    
    def _filter_by_article_models(self, models: List[str]):
        original_len = len(self.df)        
        mask = pd.Series([True] * len(self.df), index=self.df.index)
        
        for i in range(1, 5):
            model_col = f'article_{i}_model'
            if model_col in self.df.columns:
                mask &= self.df[model_col].isin(models)
        
        self.df = self.df[mask].copy()
        print(f"Filtered by article models {models}: {original_len} -> {len(self.df)} rows")
    
    def print_statistics(self):
        print("\nAsset Weighting Statistics:")
        print("-" * 60)
        print(f"Total rows: {len(self.df)}")
        print()
        
        for asset_name, column in ASSET_WEIGHT_COLUMNS.items():
            weights = self.df[column].dropna()
            print(f"{asset_name:25s} | n={len(weights):4d} | mean={weights.mean():.3f} | std={weights.std():.3f}")
    
    def plot_by_asset(
        self, 
        title: Optional[str] = None,
        bins: int = 20,
        alpha: float = 0.6,
        figsize: tuple = (12, 6),
        output_path: Optional[str] = None
    ):
        """
        Plot histograms grouped by asset type (one curve per asset).
        
        Args:
            title: Plot title
            bins: Number of histogram bins
            alpha: Transparency
            figsize: Figure size
            output_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = sns.color_palette("husl", len(ASSET_WEIGHT_COLUMNS))
        
        for (asset_name, column), color in zip(ASSET_WEIGHT_COLUMNS.items(), colors):
            weights = self.df[column].dropna()
            
            ax.hist(
                weights, 
                bins=bins, 
                alpha=alpha, 
                label=asset_name,
                color=color, 
                density=True, 
                edgecolor='black', 
                linewidth=0.5
            )
        
        ax.set_xlabel('Weighting', fontsize=12)
        ax.set_ylabel('Normalized Frequency (Density)', fontsize=12)
        inf_models = self.df['inference_model'].unique().tolist()
        news_gen_models = list(set(
            model 
            for i in range(1, 5) 
            for model in self.df[f'article_{i}_model'].dropna().unique()
        ))

        if title is None:
            title = f'Distribution of Asset Weightings by Asset Type (n={len(self.df)}) \n Inference Models: {inf_models} ; News Generation Models: {news_gen_models}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.legend(fontsize=10, loc='best')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_by_sentiment(
        self,
        title: Optional[str] = None,
        bins: int = 20,
        alpha: float = 0.6,
        figsize: tuple = (12, 6),
        output_path: Optional[str] = None
    ):
        """
        Plot histograms grouped by sentiment across all assets.
        Matches each asset's weighting with its article's sentiment.
        
        Args:
            title: Plot title
            bins: Number of histogram bins
            alpha: Transparency
            figsize: Figure size
            output_path: Path to save figure
        """
        sentiment_weight_data = []
        
        for i in range(1, 5):  # For each article position
            asset_col = f'article_{i}_asset_type'
            sentiment_col = f'article_{i}_sentiment'
            
            if asset_col in self.df.columns and sentiment_col in self.df.columns:
                for idx, row in self.df.iterrows():
                    asset_type = row[asset_col]
                    sentiment = row[sentiment_col]
                    
                    if pd.notna(asset_type) and pd.notna(sentiment):
                        weight_column = ASSET_WEIGHT_COLUMNS.get(asset_type)
                        if weight_column:
                            weight = row[weight_column]
                            if pd.notna(weight):
                                sentiment_weight_data.append({
                                    'sentiment': sentiment,
                                    'weight': weight,
                                    'asset_type': asset_type
                                })
        
        if not sentiment_weight_data:
            raise ValueError("No sentiment-weight pairs found")
        
        combined_df = pd.DataFrame(sentiment_weight_data)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sentiments = combined_df['sentiment'].unique()
        colors = sns.color_palette("Set2", len(sentiments))
        
        for sentiment, color in zip(sentiments, colors):
            sentiment_weights = combined_df[combined_df['sentiment'] == sentiment]['weight']
            
            ax.hist(
                sentiment_weights,
                bins=bins,
                alpha=alpha,
                label=f'{sentiment.capitalize()} (n={len(sentiment_weights)})',
                color=color,
                density=True,
                edgecolor='black',
                linewidth=0.5
            )
        
        ax.set_xlabel('Weighting', fontsize=12)
        ax.set_ylabel('Normalized Frequency (Density)', fontsize=12)
        
        if title is None:
            title = f'Distribution of Weightings by Article Sentiment (n={len(combined_df)})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.legend(fontsize=10, loc='best')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()

    def plot_highest_weighting_frequency(
        self,
        title: Optional[str] = None,
        figsize: tuple = (12, 6),
        output_path: Optional[str] = None
    ):
        """
        Bar chart: percentage of times each asset had the highest weighting by model.
        """
        if 'inference_model' not in self.df.columns:
            raise ValueError("No 'inference_model' column found")
        
        models = sorted(self.df['inference_model'].unique())
        assets = list(ASSET_WEIGHT_COLUMNS.keys())
        
        # Count max weightings per model
        results = {model: {asset: 0 for asset in assets} for model in models}
        
        for model in models:
            model_df = self.df[self.df['inference_model'] == model]
            
            for _, row in model_df.iterrows():
                weightings = {asset: row[col] for asset, col in ASSET_WEIGHT_COLUMNS.items() if pd.notna(row[col])}
                if weightings:
                    max_asset = max(weightings, key=weightings.get)
                    results[model][max_asset] += 1
            
            total = len(model_df)
            if total > 0:
                for asset in assets:
                    results[model][asset] = (results[model][asset] / total) * 100
        
        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(assets))
        width = 0.25
        
        colors = {'openai/gpt-5-mini': '#10a37f', 
                'anthropic/claude-3-5-haiku-20241022': '#d97757',
                'gemini/gemini-flash-lite-latest': '#4285f4'}
        
        labels = {'openai/gpt-5-mini': 'GPT-5 mini', 
                'anthropic/claude-3-5-haiku-20241022': 'Claude',
                'gemini/gemini-flash-lite-latest': 'Gemini'}
        
        for i, model in enumerate(models):
            values = [results[model][asset] for asset in assets]
            offset = (i - len(models)/2 + 0.5) * width
            
            bars = ax.bar(x + offset, values, width, 
                        label=labels.get(model, model),
                        color=colors.get(model, f'C{i}'),
                        alpha=0.8, edgecolor='black', linewidth=0.5)
            
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Asset Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_title(title or f'Highest Weighting Frequency by Model (n={len(self.df)})', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(assets, fontsize=11)
        ax.set_ylim(0, 105)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()

    def plot_mean_std_scatter(
        self,
        title: Optional[str] = None,
        figsize: tuple = (10, 8),
        output_path: Optional[str] = None
    ):
                
        if 'inference_model' not in self.df.columns:
            raise ValueError("No 'inference_model' column found")
        
        models = sorted(self.df['inference_model'].unique())
        assets = list(ASSET_WEIGHT_COLUMNS.keys())
        
        data_points = []
        for model in models:
            model_df = self.df[self.df['inference_model'] == model]
            for asset_name, column in ASSET_WEIGHT_COLUMNS.items():
                weights = model_df[column].dropna()
                if len(weights) > 0:
                    data_points.append({
                        'model': model, 'asset': asset_name,
                        'mean': weights.mean(), 'std': weights.std()
                    })
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = {'openai/gpt-5-mini': '#10a37f', 
                'anthropic/claude-3-5-haiku-20241022': '#d97757',
                'gemini/gemini-flash-lite-latest': '#4285f4'}
        markers = {'Equities': 'o', 'Corporate Bonds': 's', 
                'Government Bonds': '^', 'MBS': 'x'}
        labels = {'openai/gpt-5-mini': 'GPT-5 mini',
                'anthropic/claude-3-5-haiku-20241022': 'Claude',
                'gemini/gemini-flash-lite-latest': 'Gemini'}
        
        for point in data_points:
            ax.scatter(point['mean'], point['std'],
                    color=colors.get(point['model'], 'gray'),
                    marker=markers.get(point['asset'], 'o'),
                    s=150, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Legends
        model_handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=colors.get(m, 'gray'),
                                    markersize=10, label=labels.get(m, m),
                                    markeredgecolor='black', markeredgewidth=1)
                        for m in models]
        
        asset_handles = [plt.Line2D([0], [0], marker=markers[a], color='w',
                                    markerfacecolor='gray', markersize=10, label=a,
                                    markeredgecolor='black', markeredgewidth=1)
                        for a in assets]
        
        legend1 = ax.legend(handles=model_handles, loc='upper left', 
                        title='Model', frameon=True, fontsize=9)
        ax.add_artist(legend1)
        ax.legend(handles=asset_handles, loc='upper right',
                title='Asset', frameon=True, fontsize=9)
        
        ax.set_xlabel('Mean Weighting', fontsize=12, fontweight='bold')
        ax.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
        ax.set_title(title or f'Mean vs Std of Asset Weightings (n={len(self.df)})',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
        
   
def main():
    parser = argparse.ArgumentParser(description="Plot asset weighting distributions")
    parser.add_argument("csv_path", help="Path to results CSV file")
    parser.add_argument(
        "--plot-type",
        choices=["asset_dist", "sentiment_dist","asset_pct","asset_scatter"],
        default="asset_dist",
        help="Type of plot to generate"
    )
    parser.add_argument("--inference-model", help="Filter to specific inference model")
    parser.add_argument("--article-models", nargs="+", help="Filter to articles from specific models")
    parser.add_argument("--output", "-o", help="Output path for plot")
    parser.add_argument("--bins", type=int, default=20, help="Number of bins for histogram")
    parser.add_argument("--title", help="Custom plot title")
    
    args = parser.parse_args()
    
    print(f"Loading results from: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} rows")
    
    plotter = WeightingDistributionPlotter(
        df,
        model_inf_filter=args.inference_model,
        model_news_gen_filter=args.article_models
    )
    plotter.print_statistics()
    
    if args.plot_type == "asset_dist":
        plotter.plot_by_asset(title=args.title, bins=args.bins, output_path=args.output)
    
    elif args.plot_type == "sentiment_dist":
        plotter.plot_by_sentiment(title=args.title, bins=args.bins, output_path=args.output)
    
    elif args.plot_type == "asset_pct":
        plotter.plot_highest_weighting_frequency(title=args.title, output_path=args.output)

    elif args.plot_type == "asset_scatter":
        plotter.plot_mean_std_scatter(title=args.title, output_path=args.output)



if __name__ == "__main__":
    main()