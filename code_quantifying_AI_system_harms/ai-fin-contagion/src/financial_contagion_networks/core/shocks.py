"""
Shock scenarios for single-period financial contagion analysis.

This module defines different types of market shocks that can be applied
to bank portfolios in a single period.

Supports three shock modes:
- deterministic: All banks receive exact base shock values (no randomness)
- correlated: Shocks drawn from correlated distribution (correlation parameter)
- uncorrelated: Independent random shocks for each bank
"""

from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from financial_contagion_networks.core.assets import AssetType


class ShockMode(Enum):
    """Shock generation modes."""
    DETERMINISTIC = "deterministic"  # All banks get exact base shock
    CORRELATED = "correlated"  # Current behavior (correlation parameter)
    UNCORRELATED = "uncorrelated"  # Independent random shocks


@dataclass
class ShockScenario:
    """
    Defines a market shock scenario.

    Attributes:
        name: Scenario name
        asset_shocks: Dict mapping asset types to percentage losses (base values)
        fire_sale_intensity: How much fire sales amplify losses (0-1)
        correlation: Correlation between asset shocks (used in correlated mode)
        mode: Shock generation mode (deterministic, correlated, or uncorrelated)
        shock_volatility: Standard deviation of shocks around mean (for random modes)
        description: Text description
    """
    name: str
    asset_shocks: Dict[AssetType, float]
    fire_sale_intensity: float = 0.1
    correlation: float = 0.5
    mode: ShockMode = ShockMode.CORRELATED
    shock_volatility: float = 0.03
    description: str = ""

    def __post_init__(self):
        """Convert string mode to enum if needed."""
        if isinstance(self.mode, str):
            self.mode = ShockMode(self.mode)

    def get_shock(self, asset_type: AssetType) -> float:
        """Get base shock for a specific asset type."""
        return self.asset_shocks.get(asset_type, 0.0)


class ShockGenerator:
    """
    Generates random correlated shocks for Monte Carlo simulations.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize shock generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)

    def generate_correlated_shocks(
        self,
        correlation: float,
        mean_shock: float = -0.10,
        volatility: float = 0.05
    ) -> Dict[AssetType, float]:
        """
        Generate correlated shocks for different asset classes.

        Args:
            correlation: Correlation between asset shocks (0-1)
            mean_shock: Average shock magnitude
            volatility: Volatility of individual shocks

        Returns:
            Dict mapping asset types to shock percentages
        """
        # Generate common factor (market-wide shock)
        common_factor = self.rng.normal(mean_shock, volatility * np.sqrt(correlation))

        # Generate idiosyncratic shocks
        idio_vol = volatility * np.sqrt(1 - correlation)

        shocks = {}
        for asset_type in [AssetType.GOVERNMENT_BOND, AssetType.CORPORATE_BOND, AssetType.MORTGAGE, AssetType.STOCK]:
            idiosyncratic = self.rng.normal(0, idio_vol)
            shocks[asset_type] = common_factor + idiosyncratic

        # Government bonds less volatile
        shocks[AssetType.GOVERNMENT_BOND] *= 0.3

        # Mortgages moderate volatility (less than stocks, more than bonds)
        shocks[AssetType.MORTGAGE] *= 0.8

        # Stocks more volatile
        shocks[AssetType.STOCK] *= 1.5

        return shocks

    def generate_correlated_shocks_from_scenario(
        self,
        scenario: 'ShockScenario',
        volatility: Optional[float] = None
    ) -> Dict[AssetType, float]:
        """
        Generate shocks from scenario using the configured shock mode.

        Three modes supported:
        1. DETERMINISTIC: Returns exact base shocks (no randomness)
           - All banks receive identical shocks
           - Use for testing specific scenarios

        2. CORRELATED: Shocks drawn from correlated distribution
           - Higher correlation = assets move together more
           - At correlation=1.0, all assets deviate by same factor
           - At correlation=0.0, each asset deviates independently
           - This is the traditional model (current default)

        3. UNCORRELATED: Independent random shocks per asset
           - Each asset drawn independently around its mean
           - No correlation between assets
           - Maximum diversification benefit

        Args:
            scenario: Base scenario with shocks, mode, and parameters
            volatility: Standard deviation override (uses scenario.shock_volatility if None)

        Returns:
            Dict mapping asset types to shock percentages

        Examples:
            # Deterministic: Always returns {MORTGAGE: -0.20, STOCK: -0.15, ...}
            scenario = ShockScenario(
                name='test',
                asset_shocks={MORTGAGE: -0.20, STOCK: -0.15},
                mode=ShockMode.DETERMINISTIC
            )

            # Correlated: Returns {MORTGAGE: -0.19, STOCK: -0.14, ...} with correlation
            scenario = ShockScenario(
                name='test',
                asset_shocks={MORTGAGE: -0.20, STOCK: -0.15},
                mode=ShockMode.CORRELATED,
                correlation=0.9
            )

            # Uncorrelated: Returns {MORTGAGE: -0.21, STOCK: -0.16, ...} independently
            scenario = ShockScenario(
                name='test',
                asset_shocks={MORTGAGE: -0.20, STOCK: -0.15},
                mode=ShockMode.UNCORRELATED
            )
        """
        # Use scenario's volatility if not overridden
        vol = volatility if volatility is not None else scenario.shock_volatility

        # MODE 1: DETERMINISTIC - Return exact base shocks
        if scenario.mode == ShockMode.DETERMINISTIC:
            return dict(scenario.asset_shocks)

        # MODE 2: CORRELATED - Current behavior (correlated random shocks)
        elif scenario.mode == ShockMode.CORRELATED:
            correlation = scenario.correlation

            # Generate common factor (drives correlated movement)
            # Higher correlation = larger common factor variance
            common_factor = self.rng.normal(0, vol * np.sqrt(correlation))

            # Generate idiosyncratic shocks (asset-specific randomness)
            # Higher correlation = smaller idiosyncratic variance
            idio_vol = vol * np.sqrt(1 - correlation)

            shocks = {}
            for asset_type in [AssetType.GOVERNMENT_BOND, AssetType.CORPORATE_BOND,
                              AssetType.MORTGAGE, AssetType.STOCK]:
                # Base shock from scenario
                base_shock = scenario.asset_shocks.get(asset_type, 0.0)

                # Add correlated movement (common factor) + idiosyncratic noise
                idiosyncratic = self.rng.normal(0, idio_vol)
                randomized_shock = base_shock + common_factor + idiosyncratic

                shocks[asset_type] = randomized_shock

            return shocks

        # MODE 3: UNCORRELATED - Independent random shocks per asset
        elif scenario.mode == ShockMode.UNCORRELATED:
            shocks = {}
            for asset_type in [AssetType.GOVERNMENT_BOND, AssetType.CORPORATE_BOND,
                              AssetType.MORTGAGE, AssetType.STOCK]:
                # Base shock from scenario
                base_shock = scenario.asset_shocks.get(asset_type, 0.0)

                # Add independent noise (no correlation)
                noise = self.rng.normal(0, vol)
                randomized_shock = base_shock + noise

                shocks[asset_type] = randomized_shock

            return shocks

        else:
            raise ValueError(f"Unknown shock mode: {scenario.mode}")


# Predefined Scenarios

SCENARIOS = {
    'mild_stress': ShockScenario(
        name='Mild Market Stress',
        asset_shocks={
            AssetType.GOVERNMENT_BOND: -0.02,  # 2% loss
            AssetType.CORPORATE_BOND: -0.05,   # 5% loss
            AssetType.MORTGAGE: -0.03,         # 3% loss
            AssetType.STOCK: -0.10             # 10% loss
        },
        fire_sale_intensity=0.05,
        correlation=0.3,
        description='Mild market correction with low correlation'
    ),

    'moderate_crisis': ShockScenario(
        name='Moderate Financial Crisis',
        asset_shocks={
            AssetType.GOVERNMENT_BOND: -0.05,  # 5% loss (flight to quality less effective)
            AssetType.CORPORATE_BOND: -0.15,   # 15% loss
            AssetType.MORTGAGE: -0.12,         # 12% loss
            AssetType.STOCK: -0.25             # 25% loss
        },
        fire_sale_intensity=0.15,
        correlation=0.6,
        description='Significant market stress with high correlation'
    ),

    'severe_crisis': ShockScenario(
        name='Severe Financial Crisis',
        asset_shocks={
            AssetType.GOVERNMENT_BOND: -0.10,  # 10% loss
            AssetType.CORPORATE_BOND: -0.30,   # 30% loss
            AssetType.MORTGAGE: -0.25,         # 25% loss
            AssetType.STOCK: -0.40             # 40% loss
        },
        fire_sale_intensity=0.25,
        correlation=0.8,
        description='Severe crisis with high correlation and fire sales'
    ),

    'stock_crash': ShockScenario(
        name='Stock Market Crash',
        asset_shocks={
            AssetType.GOVERNMENT_BOND: 0.05,   # Flight to quality (bonds gain)
            AssetType.CORPORATE_BOND: -0.10,   # 10% loss
            AssetType.MORTGAGE: -0.05,         # 5% loss
            AssetType.STOCK: -0.35             # 35% crash
        },
        fire_sale_intensity=0.10,
        correlation=0.2,
        description='Stock-specific crash with flight to quality'
    ),

    'credit_crunch': ShockScenario(
        name='Credit Crunch',
        asset_shocks={
            AssetType.GOVERNMENT_BOND: 0.03,   # Flight to quality
            AssetType.CORPORATE_BOND: -0.25,   # 25% loss
            AssetType.MORTGAGE: -0.18,         # 18% loss
            AssetType.STOCK: -0.15             # 15% loss
        },
        fire_sale_intensity=0.20,
        correlation=0.7,
        description='Corporate credit crisis with high fire sales'
    ),

    'housing_crisis': ShockScenario(
        name='Housing Market Crash',
        asset_shocks={
            AssetType.GOVERNMENT_BOND: 0.02,   # Flight to quality
            AssetType.CORPORATE_BOND: -0.12,   # 12% loss
            AssetType.MORTGAGE: -0.35,         # 35% crash (2008-style)
            AssetType.STOCK: -0.20             # 20% loss
        },
        fire_sale_intensity=0.20,
        correlation=0.6,
        description='Housing/mortgage crisis (like 2008 subprime collapse)'
    ),

    'sovereign_crisis': ShockScenario(
        name='Sovereign Debt Crisis',
        asset_shocks={
            AssetType.GOVERNMENT_BOND: -0.20,  # 20% loss
            AssetType.CORPORATE_BOND: -0.15,   # 15% loss
            AssetType.MORTGAGE: -0.10,         # 10% loss
            AssetType.STOCK: -0.20             # 20% loss
        },
        fire_sale_intensity=0.15,
        correlation=0.5,
        description='Government bond crisis (sovereign debt concerns)'
    ),

    'no_shock': ShockScenario(
        name='No External Shock',
        asset_shocks={
            AssetType.GOVERNMENT_BOND: 0.0,
            AssetType.CORPORATE_BOND: 0.0,
            AssetType.MORTGAGE: 0.0,
            AssetType.STOCK: 0.0
        },
        fire_sale_intensity=0.05,
        correlation=0.0,
        description='Only interbank contagion, no market shock'
    )
}


def get_scenario(name: str) -> ShockScenario:
    """
    Get a predefined shock scenario by name.

    Args:
        name: Scenario name

    Returns:
        ShockScenario object

    Raises:
        KeyError: If scenario name not found
    """
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]


def list_scenarios() -> list[str]:
    """List all available scenario names."""
    return list(SCENARIOS.keys())
