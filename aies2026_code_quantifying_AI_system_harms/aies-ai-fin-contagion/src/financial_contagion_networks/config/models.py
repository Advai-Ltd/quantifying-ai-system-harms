"""
Configuration system for experiment management using Pydantic.

This module provides a complete configuration system for defining and running
financial contagion experiments. All experiment parameters are specified in
YAML files and loaded/validated through Pydantic models.

Key benefits of Pydantic approach:
- Automatic validation of all fields
- No silent defaults - all critical fields required
- Type checking and conversion
- Better error messages for missing/invalid fields
- Catches typos in field names
"""

from pydantic import BaseModel, Field, field_validator, model_validator, ValidationInfo, ConfigDict
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from enum import Enum
import yaml
import numpy as np


# ============================================================================
# Enums for modes
# ============================================================================

class NetworkMode(Enum):
    """Network generation modes."""
    FIXED = "fixed"  # Generate once, reuse (test ONE network)
    TEMPLATE = "template"  # Fixed structure, resample params (test ONE topology)
    STOCHASTIC = "stochastic"  # Generate new each run (test POPULATION)


# Import ShockMode from shocks module (single source of truth)
try:
    from financial_contagion_networks.core.shocks import ShockMode
except ImportError:
    # Fallback definition if shocks module not available
    class ShockMode(Enum):
        """Shock generation modes."""
        DETERMINISTIC = "deterministic"
        CORRELATED = "correlated"
        UNCORRELATED = "uncorrelated"


# ============================================================================
# Distribution configuration
# ============================================================================

class DistributionConfig(BaseModel):
    """
    Configuration for parameter distributions.

    Can represent either:
    - Fixed value: distribution='fixed', value=X
    - Random distribution: distribution='uniform'/'normal'/etc with parameters
    """

    distribution: str = Field(..., description="Distribution type: 'fixed', 'uniform', 'normal', etc.")

    # For fixed distributions
    value: Optional[float] = Field(None, description="Fixed value (if distribution='fixed')")

    # For uniform distributions
    min: Optional[float] = Field(None, description="Minimum value (for uniform distribution)")
    max: Optional[float] = Field(None, description="Maximum value (for uniform distribution)")

    # For normal distributions
    mean: Optional[float] = Field(None, description="Mean value (for normal distribution)")
    std: Optional[float] = Field(None, description="Standard deviation (for normal distribution)")

    @field_validator('distribution')
    @classmethod
    def validate_distribution_type(cls, v):
        valid_types = {'fixed', 'uniform', 'normal'}
        if v not in valid_types:
            raise ValueError(f"distribution must be one of {valid_types}, got '{v}'")
        return v

    @model_validator(mode='after')
    def validate_distribution_params(self):
        if self.distribution == 'fixed':
            if self.value is None:
                raise ValueError("distribution='fixed' requires 'value' parameter")
        elif self.distribution == 'uniform':
            if self.min is None or self.max is None:
                raise ValueError("distribution='uniform' requires 'min' and 'max' parameters")
            if self.min >= self.max:
                raise ValueError(f"min ({self.min}) must be < max ({self.max})")
        elif self.distribution == 'normal':
            if self.mean is None or self.std is None:
                raise ValueError("distribution='normal' requires 'mean' and 'std' parameters")

        return self

    def sample(self, rng: Optional[np.random.Generator] = None) -> float:
        """Sample a value from this distribution."""
        if self.distribution == 'fixed':
            return self.value
        elif self.distribution == 'uniform':
            if rng is None:
                rng = np.random.default_rng()
            return rng.uniform(self.min, self.max)
        elif self.distribution == 'normal':
            if rng is None:
                rng = np.random.default_rng()
            value = rng.normal(self.mean, self.std)

            # Apply truncation to enforce bounds (critical for Basel III 10% floor)
            if self.min is not None:
                value = max(value, self.min)
            if self.max is not None:
                value = min(value, self.max)

            return value

    model_config = ConfigDict(extra='forbid', validate_assignment=True)


# ============================================================================
# Portfolio configuration
# ============================================================================

class PortfolioConfig(BaseModel):
    """
    Portfolio allocation configuration for a bank group.

    Two modes supported:

    1. LEGACY MODE (backward compatible):
       - mortgage: DistributionConfig
       - remaining: {government_bond: 0.3, corporate_bond: 0.4, stock: 0.3}
       All banks in tier get same government_bond/corporate_bond/stock splits.

    2. HETEROGENEOUS MODE (Option F):
       - mortgage: Union[float, DistributionConfig]
       - government_bond: Union[float, DistributionConfig]
       - corporate_bond: Union[float, DistributionConfig]
       - stock: Union[float, DistributionConfig]
       Each bank samples its own portfolio composition. Values are normalized to sum to 1.0.

    Note: If both 'remaining' and individual assets are specified, individual assets take precedence.
    """

    # Individual asset allocations (Option F: Heterogeneous external assets)
    mortgage: Union[float, DistributionConfig] = Field(..., description="Mortgage allocation (as fraction)")
    government_bond: Optional[Union[float, DistributionConfig]] = Field(None, description="Government bond allocation")
    corporate_bond: Optional[Union[float, DistributionConfig]] = Field(None, description="Corporate bond allocation")
    stock: Optional[Union[float, DistributionConfig]] = Field(None, description="Stock allocation")

    # Legacy: Remaining portfolio allocation (for backward compatibility)
    remaining: Optional[Dict[str, float]] = Field(None, description="Allocation of remaining portfolio (legacy)")

    @model_validator(mode='after')
    def validate_portfolio_config(self):
        """Validate that either remaining OR individual assets are specified."""
        has_remaining = self.remaining is not None
        has_individual = (
            self.government_bond is not None or
            self.corporate_bond is not None or
            self.stock is not None
        )

        if not has_remaining and not has_individual:
            raise ValueError(
                "Either 'remaining' (legacy) or individual assets "
                "(government_bond, corporate_bond, stock) must be specified"
            )

        # Validate 'remaining' if using legacy mode
        if has_remaining and self.remaining is not None:
            total = sum(self.remaining.values())
            if not (0.99 <= total <= 1.01):
                raise ValueError(f"remaining portfolio allocations must sum to 1.0, got {total}")

            required_assets = {'government_bond', 'corporate_bond', 'stock'}
            if not required_assets.issubset(self.remaining.keys()):
                missing = required_assets - set(self.remaining.keys())
                raise ValueError(f"remaining portfolio must include {required_assets}, missing {missing}")

        # Validate individual assets if using heterogeneous mode
        if has_individual:
            # All three must be specified if using heterogeneous mode
            if not (self.government_bond is not None and
                    self.corporate_bond is not None and
                    self.stock is not None):
                raise ValueError(
                    "When using heterogeneous portfolios, all three assets "
                    "(government_bond, corporate_bond, stock) must be specified"
                )

        return self

    def is_heterogeneous(self) -> bool:
        """Check if this portfolio uses heterogeneous mode (Option F)."""
        return self.government_bond is not None

    model_config = ConfigDict(extra='forbid', validate_assignment=True)


# ============================================================================
# Bank group configuration
# ============================================================================

class BankGroupConfig(BaseModel):
    """Configuration for a group of banks (core or periphery)."""

    # Count is computed from network-level num_core_banks/num_periphery_banks
    # and injected during config loading - not specified in YAML bank group section
    count: Optional[int] = Field(None, gt=0, description="Number of banks in this group (computed from network level)")

    capital_ratio: DistributionConfig = Field(..., description="Capital ratio distribution")
    portfolio: PortfolioConfig = Field(..., description="Portfolio allocation")
    total_assets: float = Field(..., gt=0, description="Total assets per bank")
    external_assets_fraction: float = Field(..., ge=0, le=1, description="Fraction of assets that are external")

    # OPTION A: Variable interbank exposure
    # If specified, interbank_assets_fraction is sampled per bank (creates heterogeneity)
    # If not specified, calculated as (1 - external_assets_fraction) for all banks (original behavior)
    interbank_assets_fraction: Optional[Union[float, DistributionConfig]] = Field(
        None,
        description="Interbank assets as fraction of total assets (variable per bank if DistributionConfig)"
    )

    model_config = ConfigDict(extra='forbid', validate_assignment=True)


# ============================================================================
# Connectivity configuration
# ============================================================================

class ConnectivityConfig(BaseModel):
    """Network connectivity parameters."""

    # Connection probabilities
    core_to_core: float = Field(..., ge=0, le=1, description="Probability of core-core connection")
    core_to_periphery: float = Field(..., ge=0, le=1, description="Probability of core-periphery connection")
    periphery_to_core: float = Field(..., ge=0, le=1, description="Probability of periphery-core connection")
    periphery_to_periphery: float = Field(..., ge=0, le=1, description="Probability of periphery-periphery connection")

    # Exposure amount distributions
    core_to_core_exposure: DistributionConfig = Field(..., description="Core-core exposure distribution")
    core_to_periphery_exposure: DistributionConfig = Field(..., description="Core-periphery exposure distribution")
    periphery_to_core_exposure: DistributionConfig = Field(..., description="Periphery-core exposure distribution")
    periphery_to_periphery_exposure: DistributionConfig = Field(..., description="Periphery-periphery exposure distribution")

    model_config = ConfigDict(extra='forbid', validate_assignment=True)


# ============================================================================
# Metadata configuration
# ============================================================================

class MetadataConfig(BaseModel):
    """Experiment metadata and documentation - ALL REQUIRED (no defaults)."""

    # ALL fields REQUIRED - must be explicit in YAML
    experiment_id: str = Field(..., description="Unique experiment identifier")
    scenario_id: str = Field(..., description="Scenario identifier")
    scenario_name: str = Field(..., description="Human-readable scenario name")
    hypothesis: str = Field(..., description="Research hypothesis")
    description: str = Field(..., description="Detailed description")
    tags: List[str] = Field(..., description="Classification tags")

    model_config = ConfigDict(extra='forbid', validate_assignment=True)


# ============================================================================
# Network configuration
# ============================================================================

class NetworkConfig(BaseModel):
    """Network topology and bank parameters."""

    mode: str = Field(..., description="Network mode: 'fixed', 'template', or 'stochastic'")
    topology: str = Field(..., description="Network topology type")
    num_banks: int = Field(..., gt=0, description="Total number of banks")
    num_core_banks: int = Field(..., gt=0, description="Number of core banks")

    # Bank group configurations - either single groups OR subgroups (not both)
    core_banks: Optional[BankGroupConfig] = Field(None, description="Core bank parameters (single group)")
    periphery_banks: Optional[BankGroupConfig] = Field(None, description="Periphery bank parameters (single group)")

    # Bank subgroups - for heterogeneous bank types within core/periphery
    core_bank_subgroups: Optional[List[BankGroupConfig]] = Field(None, description="Core bank subgroups for heterogeneity")
    periphery_bank_subgroups: Optional[List[BankGroupConfig]] = Field(None, description="Periphery bank subgroups for heterogeneity")

    connectivity: ConnectivityConfig = Field(..., description="Network connectivity parameters")

    # Seeds REQUIRED for reproducibility (no defaults)
    structure_seed: int = Field(..., description="Seed for network structure generation")
    parameter_seed: int = Field(..., description="Seed for bank parameter sampling")

    @field_validator('num_core_banks')
    @classmethod
    def validate_num_core_banks(cls, v, info: ValidationInfo):
        if info.data.get('num_banks') and v >= info.data['num_banks']:
            raise ValueError(f"num_core_banks ({v}) must be < num_banks ({info.data['num_banks']})")
        return v

    @model_validator(mode='after')
    def validate_bank_groups(self) -> 'NetworkConfig':
        """Validate bank group configuration - must use either single groups OR subgroups."""
        # Check core banks
        has_core_single = self.core_banks is not None
        has_core_subgroups = self.core_bank_subgroups is not None and len(self.core_bank_subgroups) > 0

        if not has_core_single and not has_core_subgroups:
            raise ValueError("Either core_banks or core_bank_subgroups configuration is required")
        if has_core_single and has_core_subgroups:
            raise ValueError("Cannot specify both core_banks and core_bank_subgroups - use one or the other")

        # Check periphery banks
        has_periphery_single = self.periphery_banks is not None
        has_periphery_subgroups = self.periphery_bank_subgroups is not None and len(self.periphery_bank_subgroups) > 0

        if not has_periphery_single and not has_periphery_subgroups:
            raise ValueError("Either periphery_banks or periphery_bank_subgroups configuration is required")
        if has_periphery_single and has_periphery_subgroups:
            raise ValueError("Cannot specify both periphery_banks and periphery_bank_subgroups - use one or the other")

        return self

    def get_core_bank_groups(self) -> List[BankGroupConfig]:
        """Get list of core bank groups (handles both single group and subgroups)."""
        if self.core_bank_subgroups is not None:
            return self.core_bank_subgroups
        elif self.core_banks is not None:
            return [self.core_banks]
        else:
            raise ValueError("No core bank configuration found")

    def get_periphery_bank_groups(self) -> List[BankGroupConfig]:
        """Get list of periphery bank groups (handles both single group and subgroups)."""
        if self.periphery_bank_subgroups is not None:
            return self.periphery_bank_subgroups
        elif self.periphery_banks is not None:
            return [self.periphery_banks]
        else:
            raise ValueError("No periphery bank configuration found")

    def has_subgroups(self) -> bool:
        """Check if configuration uses bank subgroups."""
        return (self.core_bank_subgroups is not None and len(self.core_bank_subgroups) > 0) or \
               (self.periphery_bank_subgroups is not None and len(self.periphery_bank_subgroups) > 0)

    @property
    def num_periphery_banks(self) -> int:
        """Computed property: number of periphery banks."""
        return self.num_banks - self.num_core_banks

    model_config = ConfigDict(extra='forbid', validate_assignment=True)


# ============================================================================
# Shock configuration
# ============================================================================

class ShockConfig(BaseModel):
    """Shock scenario configuration - ALL REQUIRED (no defaults)."""

    mode: str = Field(..., description="Shock mode: 'deterministic', 'correlated', or 'uncorrelated'")

    # Asset shocks - REQUIRED, no defaults
    asset_shocks: Dict[str, float] = Field(..., description="Shock magnitudes by asset class")

    # Correlation and volatility - REQUIRED
    correlation: float = Field(..., ge=0, le=1, description="Shock correlation coefficient")
    fire_sale_intensity: float = Field(..., ge=0, description="Fire sale price impact multiplier (>=0, values >1 represent extreme crisis scenarios)")
    shock_volatility: float = Field(..., ge=0, description="Shock volatility parameter")

    @field_validator('mode')
    @classmethod
    def validate_shock_mode(cls, v):
        valid_modes = {'deterministic', 'correlated', 'uncorrelated'}
        if v not in valid_modes:
            raise ValueError(f"shock mode must be one of {valid_modes}, got '{v}'")
        return v

    @field_validator('asset_shocks')
    @classmethod
    def validate_asset_shocks(cls, v):
        required_assets = {'government_bond', 'corporate_bond', 'mortgage', 'stock'}
        if not required_assets.issubset(v.keys()):
            missing = required_assets - set(v.keys())
            raise ValueError(f"asset_shocks must include {required_assets}, missing {missing}")
        return v

    model_config = ConfigDict(extra='forbid', validate_assignment=True)


# ============================================================================
# Simulation configuration
# ============================================================================

class SimulationConfig(BaseModel):
    """Monte Carlo simulation parameters - ALL REQUIRED (no defaults)."""

    # ALL fields REQUIRED - no defaults, no optionals
    num_runs: int = Field(..., gt=0, description="Number of Monte Carlo runs")
    seed: int = Field(..., description="Master random seed for reproducibility")
    fire_sales_enabled: bool = Field(..., description="Enable fire sale contagion mechanism")
    use_priority_claims: bool = Field(..., description="Interbank creditors junior to external (Option 4)")

    model_config = ConfigDict(extra='forbid', validate_assignment=True)


# ============================================================================
# Output configuration
# ============================================================================

class OutputConfig(BaseModel):
    """Output and results configuration - ALL REQUIRED (no defaults)."""

    # ALL fields REQUIRED - must be explicit in YAML
    output_dir: str = Field(..., description="Output directory path")
    save_summary: bool = Field(..., description="Save summary statistics")
    save_detailed_results: bool = Field(..., description="Save detailed simulation results")
    save_network_snapshots: bool = Field(..., description="Save network state snapshots")
    save_config_copy: bool = Field(..., description="Save copy of configuration")
    generate_plots: bool = Field(..., description="Generate visualization plots")
    plot_formats: List[str] = Field(..., description="Plot file formats (e.g., ['png', 'pdf'])")
    verbose: bool = Field(..., description="Verbose output during execution")

    model_config = ConfigDict(extra='forbid', validate_assignment=True)


# ============================================================================
# Top-level experiment configuration
# ============================================================================

class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""

    metadata: MetadataConfig = Field(..., description="Experiment metadata")
    network: NetworkConfig = Field(..., description="Network configuration")
    shock: ShockConfig = Field(..., description="Shock scenario configuration")
    simulation: SimulationConfig = Field(..., description="Simulation parameters")
    output: OutputConfig = Field(..., description="Output configuration")

    @model_validator(mode='after')
    def validate_bank_counts(self) -> 'ExperimentConfig':
        """Validate bank counts are consistent."""
        network = self.network

        # Check core banks count matches num_core_banks
        if network.core_banks and network.core_banks.count:
            if network.core_banks.count != network.num_core_banks:
                raise ValueError(
                    f"Core banks count mismatch: "
                    f"network.num_core_banks={network.num_core_banks} but "
                    f"core_banks.count={network.core_banks.count}"
                )

        # Check core bank subgroups counts sum to num_core_banks
        if network.core_bank_subgroups:
            actual_core_count = sum(group.count for group in network.core_bank_subgroups if group.count)
            if actual_core_count != network.num_core_banks:
                raise ValueError(
                    f"Core banks count mismatch: "
                    f"network.num_core_banks={network.num_core_banks} but "
                    f"subgroups sum to {actual_core_count}"
                )

        # Check periphery banks count
        if network.periphery_banks and network.periphery_banks.count:
            expected_periphery = network.num_banks - network.num_core_banks
            if network.periphery_banks.count != expected_periphery:
                raise ValueError(
                    f"Periphery banks count mismatch: "
                    f"expected {expected_periphery} (num_banks - num_core_banks) but "
                    f"periphery_banks.count={network.periphery_banks.count}"
                )

        # Check periphery bank subgroups counts sum to (num_banks - num_core_banks)
        if network.periphery_bank_subgroups:
            expected_periphery = network.num_banks - network.num_core_banks
            actual_periphery_count = sum(group.count for group in network.periphery_bank_subgroups if group.count)
            if actual_periphery_count != expected_periphery:
                raise ValueError(
                    f"Periphery banks count mismatch: "
                    f"expected {expected_periphery} (num_banks - num_core_banks) but "
                    f"subgroups sum to {actual_periphery_count}"
                )

        return self

    def validate(self):
        """
        Validate complete configuration.

        Pydantic automatically validates on construction, but this method
        is kept for backwards compatibility and to re-run validation after
        manual field modifications.
        """
        # Re-run bank count validation to check consistency after manual changes
        network = self.network

        # Check core banks count matches num_core_banks
        if network.core_banks and network.core_banks.count:
            if network.core_banks.count != network.num_core_banks:
                raise ValueError(
                    f"Core banks count mismatch: "
                    f"network.num_core_banks={network.num_core_banks} but "
                    f"core_banks.count={network.core_banks.count}"
                )

        # Check periphery banks count
        if network.periphery_banks and network.periphery_banks.count:
            expected_periphery = network.num_banks - network.num_core_banks
            if network.periphery_banks.count != expected_periphery:
                raise ValueError(
                    f"Periphery banks count mismatch: "
                    f"expected {expected_periphery} (num_banks - num_core_banks) but "
                    f"periphery_banks.count={network.periphery_banks.count}"
                )

    model_config = ConfigDict(extra='forbid', validate_assignment=True)


# ============================================================================
# Config loading functions
# ============================================================================

def load_config(config_path: Union[str, Path]) -> ExperimentConfig:
    """
    Load configuration from YAML file using Pydantic validation.

    This replaces the old ConfigLoader class with Pydantic's automatic
    validation. Benefits:
    - No manual parsing code
    - Automatic type validation
    - Missing required fields caught immediately
    - Unknown fields (typos) caught immediately
    - Better error messages

    Args:
        config_path: Path to YAML configuration file

    Returns:
        ExperimentConfig instance with all validation completed

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config is invalid (missing fields, wrong types, etc.)
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    # Pre-process network section to inject count values into bank groups
    if 'network' in data:
        network_data = data['network']

        # Inject count into core_banks from network-level num_core_banks
        if 'core_banks' in network_data and 'num_core_banks' in network_data:
            network_data['core_banks']['count'] = network_data['num_core_banks']

        # Inject count into periphery_banks (computed as num_banks - num_core_banks)
        if 'periphery_banks' in network_data and 'num_banks' in network_data and 'num_core_banks' in network_data:
            periphery_count = network_data['num_banks'] - network_data['num_core_banks']
            network_data['periphery_banks']['count'] = periphery_count

        # Remove num_periphery_banks if present (it's computed, not a schema field)
        network_data.pop('num_periphery_banks', None)

    # Pydantic handles all validation automatically!
    # No manual parsing, no missing fields, no silent defaults
    return ExperimentConfig(**data)


# Backwards compatibility: keep ConfigLoader name
class ConfigLoader:
    """
    Backwards compatibility wrapper for load_config function.

    The old ConfigLoader class with manual parsing has been replaced
    with Pydantic's automatic validation. This class provides the
    same interface for backwards compatibility.
    """

    @staticmethod
    def load(config_path: Union[str, Path]) -> ExperimentConfig:
        """Load configuration from YAML file."""
        return load_config(config_path)
