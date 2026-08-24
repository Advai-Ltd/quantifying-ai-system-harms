"""
Network generators for config-driven experiment system.

Supports three modes:
- fixed: Generate once, reuse (test ONE specific network)
- template: Fixed structure, resample parameters (test ONE topology)
- stochastic: Generate new network each run (test POPULATION)
"""

from typing import Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from financial_contagion_networks.core.network import ContagionNetwork
from financial_contagion_networks.core.assets import AssetType, Portfolio, STANDARD_RISK_WEIGHTS
from financial_contagion_networks.core.bank import Bank
from financial_contagion_networks.config import (
    NetworkConfig,
    BankGroupConfig,
    PortfolioConfig,
    DistributionConfig,
)


# ============================================================================
# Utility: Create Banks with Portfolios
# ============================================================================

def create_portfolio_bank(
    bank_id: int,
    total_assets: float,
    capital_ratio: float,
    portfolio_weights: Optional[Dict[AssetType, float]] = None,
    interbank_fraction: float = 0.15
) -> Bank:
    """
    Create a bank with a diversified portfolio.

    Args:
        bank_id: Bank identifier
        total_assets: Total assets
        capital_ratio: Target capital ratio (equity/assets)
        portfolio_weights: Dict mapping asset types to portfolio weights (must sum to 1.0)
        interbank_fraction: Fraction of assets in interbank loans

    Returns:
        Bank with portfolio
    """
    # Default portfolio: 30% gov bonds, 25% corp bonds, 25% mortgages, 20% stocks
    if portfolio_weights is None:
        portfolio_weights = {
            AssetType.GOVERNMENT_BOND: 0.30,
            AssetType.CORPORATE_BOND: 0.25,
            AssetType.MORTGAGE: 0.25,
            AssetType.STOCK: 0.20
        }

    # Validate weights
    total_weight = sum(portfolio_weights.values())
    if abs(total_weight - 1.0) > 0.01:
        raise ValueError(f"Portfolio weights must sum to 1.0, got {total_weight}")

    # Calculate asset allocation
    external_assets = total_assets * (1.0 - interbank_fraction)
    # Pre-allocate interbank assets to match target fraction
    # Network generation will redistribute these via exposures
    interbank_assets = total_assets * interbank_fraction

    # Calculate liabilities
    equity = total_assets * capital_ratio
    total_liabilities = total_assets - equity

    # CRITICAL FIX: Pre-allocate interbank liabilities to balance interbank assets
    # This prevents capital inflation when network connections are added
    interbank_liabilities = total_assets * interbank_fraction
    external_liabilities = total_liabilities - interbank_liabilities

    # Create portfolio
    portfolio = Portfolio()
    for asset_type, weight in portfolio_weights.items():
        amount = external_assets * weight
        risk_weight = STANDARD_RISK_WEIGHTS.get(asset_type, 1.0)
        portfolio.add_asset_class(asset_type, amount, risk_weight)

    # Create bank with portfolio
    bank = Bank(
        bank_id=bank_id,
        external_assets=external_assets,
        interbank_assets=interbank_assets,
        interbank_liabilities=interbank_liabilities,
        external_liabilities=external_liabilities,
        portfolio=portfolio
    )

    return bank


# ============================================================================
# Network Template (for template mode)
# ============================================================================

@dataclass
class NetworkTopology:
    """
    Stores fixed network topology for template mode.

    In template mode, the topology (structure) is fixed but parameters
    (capital ratios, portfolio weights, exposure amounts) are resampled
    each time the network is generated.
    """
    # Structure information
    num_banks: int
    num_core_banks: int

    # Interbank connections (list of (from_id, to_id) tuples)
    connections: list[Tuple[int, int]]

    def __post_init__(self):
        """Validate topology."""
        if self.num_core_banks > self.num_banks:
            raise ValueError(
                f"num_core_banks ({self.num_core_banks}) > num_banks ({self.num_banks})"
            )


class NetworkTemplate:
    """
    Represents a fixed network topology that can be instantiated multiple times
    with different parameters.

    Used in template mode: structure is fixed, parameters vary.
    """

    def __init__(self, config: NetworkConfig, topology: NetworkTopology):
        """
        Initialize network template.

        Args:
            config: Network configuration
            topology: Fixed network topology
        """
        self.config = config
        self.topology = topology

    def generate(self, parameter_seed: Optional[int] = None) -> ContagionNetwork:
        """
        Generate a network instance from this template.

        Args:
            parameter_seed: Seed for parameter sampling (None = use config seed)

        Returns:
            ContagionNetwork instance
        """
        # Use provided seed or fall back to config
        seed = parameter_seed if parameter_seed is not None else self.config.parameter_seed

        # Generate network with fixed structure but resampled parameters
        return NetworkGenerator._generate_network_with_topology(
            config=self.config,
            topology=self.topology,
            parameter_seed=seed
        )


# ============================================================================
# Main Network Generator
# ============================================================================

class NetworkGenerator:
    """
    Universal network generator supporting all three modes.

    Usage:
        # Stochastic mode (new network each time)
        config = load_config('config.yaml')
        network = NetworkGenerator.from_config(config.network, run_id=0)

        # Template mode (fixed structure, vary parameters)
        template = NetworkGenerator.create_template(config.network)
        network1 = template.generate(parameter_seed=42)
        network2 = template.generate(parameter_seed=43)

        # Fixed mode (identical network every time)
        network = NetworkGenerator.from_config(config.network)
        # Always returns same network
    """

    @staticmethod
    def from_config(
        config: NetworkConfig,
        run_id: int = 0,
        template: Optional[NetworkTemplate] = None
    ) -> ContagionNetwork:
        """
        Generate a network from configuration.

        Args:
            config: Network configuration
            run_id: Run number (used for seed in stochastic mode)
            template: Pre-generated template (for template mode)

        Returns:
            ContagionNetwork instance
        """
        if config.mode == 'fixed':
            # Fixed mode: Always generate same network
            return NetworkGenerator._generate_fixed_network(config)

        elif config.mode == 'template':
            # Template mode: Use template if provided, otherwise create one
            if template is None:
                template = NetworkGenerator.create_template(config)

            # Generate instance with run-specific parameter seed
            param_seed = None
            if config.parameter_seed is not None:
                param_seed = config.parameter_seed + run_id

            return template.generate(parameter_seed=param_seed)

        elif config.mode == 'stochastic':
            # Stochastic mode: New network each time
            structure_seed = None
            parameter_seed = None

            if config.structure_seed is not None:
                structure_seed = config.structure_seed + run_id
            if config.parameter_seed is not None:
                parameter_seed = config.parameter_seed + run_id

            return NetworkGenerator._generate_stochastic_network(
                config=config,
                structure_seed=structure_seed,
                parameter_seed=parameter_seed
            )

        else:
            raise ValueError(f"Unknown network mode: {config.mode}")

    @staticmethod
    def create_template(config: NetworkConfig) -> NetworkTemplate:
        """
        Create a network template for template mode.

        Args:
            config: Network configuration (must have mode=template)

        Returns:
            NetworkTemplate instance
        """
        if config.mode != 'template':
            raise ValueError(f"Can only create template for template mode, got {config.mode}")

        # Generate topology once with structure seed
        topology = NetworkGenerator._generate_topology(config, config.structure_seed)

        return NetworkTemplate(config, topology)

    # ========================================================================
    # Internal generation methods
    # ========================================================================

    @staticmethod
    def _generate_fixed_network(config: NetworkConfig) -> ContagionNetwork:
        """Generate network for fixed mode (always identical)."""
        # Use both seeds for complete reproducibility
        topology = NetworkGenerator._generate_topology(config, config.structure_seed)
        return NetworkGenerator._generate_network_with_topology(
            config=config,
            topology=topology,
            parameter_seed=config.parameter_seed
        )

    @staticmethod
    def _generate_stochastic_network(
        config: NetworkConfig,
        structure_seed: Optional[int],
        parameter_seed: Optional[int]
    ) -> ContagionNetwork:
        """Generate network for stochastic mode (new each time)."""
        # OPTION D: Use parameter_seed for topology generation too
        # This creates more variation in network structure across runs
        # Instead of deterministic topology per run_id, we get variation
        topology_seed = parameter_seed if parameter_seed is not None else structure_seed
        topology = NetworkGenerator._generate_topology(config, topology_seed)
        return NetworkGenerator._generate_network_with_topology(
            config=config,
            topology=topology,
            parameter_seed=parameter_seed
        )

    @staticmethod
    def _generate_topology(
        config: NetworkConfig,
        seed: Optional[int]
    ) -> NetworkTopology:
        """
        Generate network topology (structure only, no parameters).

        Args:
            config: Network configuration
            seed: Random seed for structure generation

        Returns:
            NetworkTopology with fixed connections
        """
        rng = np.random.RandomState(seed)

        num_core = config.num_core_banks
        num_periphery = config.num_banks - num_core
        num_banks = config.num_banks
        connectivity = config.connectivity

        # Generate interbank connections based on probabilities
        connections = []

        # Core to core
        for i in range(num_core):
            for j in range(num_core):
                if i != j and rng.random() < connectivity.core_to_core:
                    connections.append((i, j))

        # Core to periphery
        for i in range(num_core):
            for j in range(num_core, num_banks):
                if rng.random() < connectivity.core_to_periphery:
                    connections.append((i, j))

        # Periphery to core
        for i in range(num_core, num_banks):
            for j in range(num_core):
                if rng.random() < connectivity.periphery_to_core:
                    connections.append((i, j))

        # Periphery to periphery
        for i in range(num_core, num_banks):
            for j in range(num_core, num_banks):
                if i != j and rng.random() < connectivity.periphery_to_periphery:
                    connections.append((i, j))

        return NetworkTopology(
            num_banks=num_banks,
            num_core_banks=num_core,
            connections=connections
        )

    @staticmethod
    def _generate_network_with_topology(
        config: NetworkConfig,
        topology: NetworkTopology,
        parameter_seed: Optional[int]
    ) -> ContagionNetwork:
        """
        Generate complete network with given topology and sampled parameters.

        Args:
            config: Network configuration
            topology: Fixed network topology
            parameter_seed: Seed for parameter sampling

        Returns:
            Complete ContagionNetwork
        """
        rng = np.random.RandomState(parameter_seed)
        network = ContagionNetwork()

        num_core = topology.num_core_banks
        num_banks = topology.num_banks

        # Create core banks (handles both single group and subgroups)
        bank_id = 0
        core_groups = config.get_core_bank_groups()
        for group_config in core_groups:
            group_count = group_config.count if group_config.count else config.num_core_banks
            for _ in range(group_count):
                bank = NetworkGenerator._create_bank(
                    bank_id=bank_id,
                    group_config=group_config,
                    rng=rng
                )
                network.add_bank(bank)
                bank_id += 1

        # Create periphery banks (handles both single group and subgroups)
        periphery_groups = config.get_periphery_bank_groups()
        for group_config in periphery_groups:
            group_count = group_config.count if group_config.count else (config.num_banks - config.num_core_banks)
            for _ in range(group_count):
                bank = NetworkGenerator._create_bank(
                    bank_id=bank_id,
                    group_config=group_config,
                    rng=rng
                )
                network.add_bank(bank)
                bank_id += 1

        # Add interbank exposures using fixed topology
        connectivity = config.connectivity

        for from_id, to_id in topology.connections:
            # Determine which exposure distribution to use
            from_is_core = from_id < num_core
            to_is_core = to_id < num_core

            if from_is_core and to_is_core:
                exposure_dist = connectivity.core_to_core_exposure
            elif from_is_core and not to_is_core:
                exposure_dist = connectivity.core_to_periphery_exposure
            elif not from_is_core and to_is_core:
                exposure_dist = connectivity.periphery_to_core_exposure
            else:
                exposure_dist = connectivity.periphery_to_periphery_exposure

            # Sample exposure amount
            if isinstance(exposure_dist, DistributionConfig):
                exposure = exposure_dist.sample(rng)
            elif isinstance(exposure_dist, (float, int)):
                exposure = float(exposure_dist)
            else:
                raise ValueError(f"Invalid exposure distribution: {exposure_dist}")

            network.add_exposure(from_id, to_id, exposure)

        return network

    @staticmethod
    def _create_bank(
        bank_id: int,
        group_config: BankGroupConfig,
        rng: np.random.RandomState
    ) -> Bank:
        """
        Create a single bank from group configuration.

        Args:
            bank_id: Bank identifier
            group_config: Configuration for this bank group
            rng: Random number generator

        Returns:
            Bank instance
        """
        # Sample capital ratio
        if isinstance(group_config.capital_ratio, DistributionConfig):
            capital_ratio = group_config.capital_ratio.sample(rng)
        elif isinstance(group_config.capital_ratio, (float, int)):
            capital_ratio = float(group_config.capital_ratio)
        else:
            raise ValueError(f"Invalid capital ratio: {group_config.capital_ratio}")

        # Sample total assets
        if isinstance(group_config.total_assets, DistributionConfig):
            total_assets = group_config.total_assets.sample(rng)
        elif isinstance(group_config.total_assets, (float, int)):
            total_assets = float(group_config.total_assets)
        else:
            raise ValueError(f"Invalid total assets: {group_config.total_assets}")

        # Calculate portfolio weights
        portfolio_weights = NetworkGenerator._calculate_portfolio_weights(
            group_config.portfolio,
            rng
        )

        # OPTION A: Variable interbank exposure
        # Check if interbank_assets_fraction is specified (new variable exposure)
        if group_config.interbank_assets_fraction is not None:
            # Variable interbank fraction per bank (heterogeneity!)
            ib_fraction_config = group_config.interbank_assets_fraction
            if isinstance(ib_fraction_config, DistributionConfig):
                interbank_fraction = ib_fraction_config.sample(rng)  # type: ignore[arg-type]
            elif isinstance(ib_fraction_config, (float, int)):
                interbank_fraction = float(ib_fraction_config)
            else:
                raise ValueError(f"Invalid interbank_assets_fraction: {ib_fraction_config}")
        else:
            # Original behavior: fixed fraction calculated from external_assets_fraction
            interbank_fraction = 1.0 - group_config.external_assets_fraction

        # Create bank using existing function
        bank = create_portfolio_bank(
            bank_id=bank_id,
            total_assets=total_assets,
            capital_ratio=capital_ratio,
            portfolio_weights=portfolio_weights,
            interbank_fraction=interbank_fraction
        )

        return bank

    @staticmethod
    def _calculate_portfolio_weights(
        portfolio_config: PortfolioConfig,
        rng: np.random.RandomState
    ) -> Dict[AssetType, float]:
        """
        Calculate portfolio weights from configuration.

        Supports partial specification with 'remaining' allocation.
        When using distributions, must specify 'remaining' to ensure sum = 1.0.

        Args:
            portfolio_config: Portfolio configuration
            rng: Random number generator

        Returns:
            Dictionary mapping AssetType to weights (sum = 1.0)

        Raises:
            ValueError: If weights don't sum to 1.0
            ValueError: If asset specified both explicitly and in remaining
        """
        weights = {}
        total_allocated = 0.0

        # Asset type mapping
        asset_map = {
            'government_bond': AssetType.GOVERNMENT_BOND,
            'corporate_bond': AssetType.CORPORATE_BOND,
            'mortgage': AssetType.MORTGAGE,
            'stock': AssetType.STOCK
        }

        # OPTION F: Heterogeneous external assets
        # If portfolio has individual asset specs, sample each one and normalize
        if portfolio_config.is_heterogeneous():
            # Sample each asset individually
            raw_weights = {}

            for field_name, asset_type in asset_map.items():
                # Get config for this asset
                asset_config = getattr(portfolio_config, field_name, None)
                if asset_config is None:
                    raise ValueError(f"Missing asset config for {field_name} in heterogeneous mode")

                # Sample the weight
                if isinstance(asset_config, DistributionConfig):
                    raw_weights[asset_type] = asset_config.sample(rng)  # type: ignore[arg-type]
                elif isinstance(asset_config, (float, int)):
                    raw_weights[asset_type] = float(asset_config)
                else:
                    raise ValueError(f"Invalid asset config type for {field_name}: {type(asset_config)}")

            # Normalize to sum to 1.0 (simple and realistic)
            total = sum(raw_weights.values())
            if total <= 0:
                raise ValueError(f"Total portfolio weight is {total}, must be positive")

            for asset_type, raw_weight in raw_weights.items():
                weights[asset_type] = raw_weight / total

        else:
            # LEGACY MODE: mortgage + remaining dict
            # Sample mortgage allocation
            mortgage_config = portfolio_config.mortgage
            if isinstance(mortgage_config, DistributionConfig):
                mortgage_weight = mortgage_config.sample(rng)  # type: ignore[arg-type]
            elif isinstance(mortgage_config, (float, int)):
                mortgage_weight = float(mortgage_config)
            else:
                raise ValueError(f"Invalid mortgage config type: {type(mortgage_config)}")

            weights[AssetType.MORTGAGE] = mortgage_weight
            total_allocated = mortgage_weight

            # Validate remaining amount is non-negative
            remaining_amount = 1.0 - total_allocated
            if remaining_amount < 0:
                raise ValueError(
                    f"Mortgage weight {mortgage_weight:.4f} exceeds 1.0, "
                    f"no room for remaining assets"
                )

            # Distribute remaining amount according to remaining specification
            # If remaining_amount is 0, remaining assets will have 0 weight
            if portfolio_config.remaining and remaining_amount > 0:
                for field_name, relative_weight in portfolio_config.remaining.items():
                    asset_type = asset_map.get(field_name)
                    if asset_type is None:
                        raise ValueError(f"Unknown asset type in remaining: {field_name}")

                    if asset_type in weights:
                        raise ValueError(
                            f"Asset {field_name} specified both in mortgage and remaining"
                        )

                    weights[asset_type] = remaining_amount * relative_weight
            elif portfolio_config.remaining and remaining_amount == 0:
                # Mortgage takes 100%, remaining assets get 0 weight
                for field_name in portfolio_config.remaining.keys():
                    asset_type = asset_map.get(field_name)
                    if asset_type:
                        weights[asset_type] = 0.0

        # Validate weights sum to 1.0
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Portfolio weights do not sum to 1.0: {total}\n"
                f"Weights: {weights}"
            )

        return weights


# ============================================================================
# Convenience functions
# ============================================================================

def generate_network(
    config: NetworkConfig,
    run_id: int = 0,
    template: Optional[NetworkTemplate] = None
) -> ContagionNetwork:
    """
    Convenience function to generate a network.

    Args:
        config: Network configuration
        run_id: Run number (for stochastic/template modes)
        template: Pre-generated template (for template mode)

    Returns:
        ContagionNetwork instance
    """
    return NetworkGenerator.from_config(config, run_id, template)
