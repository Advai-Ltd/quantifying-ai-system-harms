"""
Unit tests for configuration system.

Tests all components of the config module including:
- DistributionConfig (fixed values and distributions)
- Enums (NetworkMode, ShockMode)
- Configuration dataclasses
- ConfigLoader (YAML parsing and validation)
"""

import pytest
import numpy as np
import tempfile
import yaml
from pathlib import Path

from financial_contagion_networks.config import (
    DistributionConfig,
    NetworkMode,
    ShockMode,
    MetadataConfig,
    NetworkConfig,
    ShockConfig,
    SimulationConfig,
    OutputConfig,
    ExperimentConfig,
    PortfolioConfig,
    BankGroupConfig,
    ConnectivityConfig,
    ConfigLoader,
    load_config
)


# ============================================================================
# Test DistributionConfig
# ============================================================================

class TestDistributionConfig:
    """Test DistributionConfig class."""

    def test_fixed_value(self):
        """Test fixed value creation and sampling."""
        dist = DistributionConfig(distribution='fixed', value=0.15)
        assert dist.distribution == 'fixed'

        rng = np.random.RandomState(42)
        assert dist.sample(rng) == 0.15
        assert dist.sample(rng) == 0.15  # Should always return same value

    def test_uniform_distribution(self):
        """Test uniform distribution sampling."""
        dist = DistributionConfig(distribution='uniform', min=0.1, max=0.2)
        assert dist.distribution != 'fixed'

        rng = np.random.RandomState(42)
        samples = [dist.sample(rng) for _ in range(100)]

        # Check all samples are in range
        assert all(0.1 <= s <= 0.2 for s in samples)

        # Check we get different values
        assert len(set(samples)) > 10

    def test_normal_distribution(self):
        """Test normal distribution sampling."""
        dist = DistributionConfig(distribution='normal', mean=0.15, std=0.02)
        assert dist.distribution != 'fixed'

        rng = np.random.RandomState(42)
        samples = [dist.sample(rng) for _ in range(1000)]

        # Check mean and std are approximately correct
        assert abs(np.mean(samples) - 0.15) < 0.01
        assert abs(np.std(samples) - 0.02) < 0.01

    def test_uniform_missing_params(self):
        """Test uniform distribution with missing parameters - Pydantic validates at construction."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="uniform.*requires.*min.*max"):
            DistributionConfig(distribution='uniform', min=0.1)  # Missing max

    def test_normal_missing_params(self):
        """Test normal distribution with missing parameters - Pydantic validates at construction."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="normal.*requires.*mean.*std"):
            DistributionConfig(distribution='normal', mean=0.15)  # Missing std

    def test_unknown_distribution(self):
        """Test unknown distribution type - Pydantic validates at construction."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="distribution must be one of"):
            DistributionConfig(distribution='exponential', mean=0.15)


# ============================================================================
# Test Enums
# ============================================================================

class TestEnums:
    """Test enum classes."""

    def test_network_mode_values(self):
        """Test NetworkMode enum values."""
        assert NetworkMode.FIXED.value == "fixed"
        assert NetworkMode.TEMPLATE.value == "template"
        assert NetworkMode.STOCHASTIC.value == "stochastic"

    def test_network_mode_from_string(self):
        """Test creating NetworkMode from string."""
        assert NetworkMode("fixed") == NetworkMode.FIXED
        assert NetworkMode("template") == NetworkMode.TEMPLATE
        assert NetworkMode("stochastic") == NetworkMode.STOCHASTIC

    def test_shock_mode_values(self):
        """Test ShockMode enum values."""
        assert ShockMode.DETERMINISTIC.value == "deterministic"
        assert ShockMode.CORRELATED.value == "correlated"
        assert ShockMode.UNCORRELATED.value == "uncorrelated"

    def test_shock_mode_from_string(self):
        """Test creating ShockMode from string."""
        assert ShockMode("deterministic") == ShockMode.DETERMINISTIC
        assert ShockMode("correlated") == ShockMode.CORRELATED
        assert ShockMode("uncorrelated") == ShockMode.UNCORRELATED


# ============================================================================
# Test Configuration Dataclasses
# ============================================================================

class TestMetadataConfig:
    """Test MetadataConfig."""

    def test_minimal_metadata(self):
        """Test minimal metadata configuration - all fields required with zero-defaults."""
        config = MetadataConfig(
            experiment_id="E1",
            scenario_id="E1.1",
            scenario_name="Test Scenario",
            hypothesis="Test hypothesis",
            description="Test description",
            tags=[]
        )
        assert config.experiment_id == "E1"
        assert config.scenario_id == "E1.1"
        assert config.scenario_name == "Test Scenario"
        assert config.hypothesis == "Test hypothesis"
        assert config.description == "Test description"
        assert config.tags == []

    def test_full_metadata(self):
        """Test full metadata configuration."""
        config = MetadataConfig(
            experiment_id="E1",
            scenario_id="E1.1",
            scenario_name="Test Scenario",
            hypothesis="Test hypothesis",
            description="Test description",
            tags=["test", "correlation"]
        )
        assert config.hypothesis == "Test hypothesis"
        assert config.description == "Test description"
        assert config.tags == ["test", "correlation"]


class TestShockConfig:
    """Test ShockConfig."""

    def test_valid_shock_config(self):
        """Test valid shock configuration - all fields required."""
        config = ShockConfig(
            mode=ShockMode.CORRELATED,
            asset_shocks={
                'mortgage': -0.20,
                'stock': -0.15,
                'government_bond': 0.02,
                'corporate_bond': -0.10
            },
            correlation=0.8,
            fire_sale_intensity=0.25,
            shock_volatility=0.05
        )
        assert config.mode == 'correlated'
        assert config.correlation == 0.8
        assert config.fire_sale_intensity == 0.25
        assert config.shock_volatility == 0.05

    def test_shock_config_string_mode(self):
        """Test shock config with string mode - all fields required."""
        config = ShockConfig(
            mode="correlated",
            asset_shocks={
                'mortgage': -0.20,
                'stock': -0.15,
                'government_bond': 0.02,
                'corporate_bond': -0.10
            },
            correlation=0.6,
            fire_sale_intensity=0.15,
            shock_volatility=0.03
        )
        assert config.mode == 'correlated'

    def test_invalid_correlation(self):
        """Test shock config with invalid correlation - Pydantic validates at construction."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            ShockConfig(
                mode=ShockMode.CORRELATED,
                asset_shocks={
                    'mortgage': -0.20,
                    'stock': -0.15,
                    'government_bond': 0.02,
                    'corporate_bond': -0.10
                },
                correlation=1.5,
                fire_sale_intensity=0.15,
                shock_volatility=0.03
            )

    def test_negative_correlation(self):
        """Test shock config with negative correlation - Pydantic validates at construction."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            ShockConfig(
                mode=ShockMode.CORRELATED,
                asset_shocks={
                    'mortgage': -0.20,
                    'stock': -0.15,
                    'government_bond': 0.02,
                    'corporate_bond': -0.10
                },
                correlation=-0.1,
                fire_sale_intensity=0.15,
                shock_volatility=0.03
            )

    def test_invalid_fire_sale_intensity(self):
        """Test shock config with invalid fire sale intensity - Pydantic validates at construction."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="less than or equal to 10"):
            ShockConfig(
                mode=ShockMode.CORRELATED,
                asset_shocks={
                    'mortgage': -0.20,
                    'stock': -0.15,
                    'government_bond': 0.02,
                    'corporate_bond': -0.10
                },
                correlation=0.6,
                fire_sale_intensity=11.0,
                shock_volatility=0.03
            )


class TestSimulationConfig:
    """Test SimulationConfig."""

    def test_valid_simulation_config(self):
        """Test valid simulation configuration."""
        config = SimulationConfig(
            num_runs=200,
            seed=42,
            fire_sales_enabled=True,
            use_priority_claims=False
        )
        assert config.num_runs == 200
        assert config.seed == 42
        assert config.fire_sales_enabled is True
        assert config.use_priority_claims is False

    def test_invalid_num_runs(self):
        """Test simulation config with invalid num_runs."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SimulationConfig(
                num_runs=0,
                seed=42,
                fire_sales_enabled=True,
                use_priority_claims=False
            )

    def test_negative_num_runs(self):
        """Test simulation config with negative num_runs."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SimulationConfig(
                num_runs=-10,
                seed=42,
                fire_sales_enabled=True,
                use_priority_claims=False
            )


class TestNetworkConfig:
    """Test NetworkConfig."""

    def test_network_config_requires_bank_groups(self):
        """Test that network config requires bank group configurations - Pydantic enforces required fields."""
        from pydantic import ValidationError
        # Error raised during initialization if no bank groups provided
        with pytest.raises(ValidationError, match="Field required"):
            NetworkConfig(
                mode=NetworkMode.STOCHASTIC,
                topology="post_2008_reformed",
                num_banks=20,
                num_core_banks=5
                # Missing core_banks, periphery_banks, connectivity, seeds - all required
            )


class TestExperimentConfig:
    """Test ExperimentConfig validation."""

    def create_minimal_config(self) -> ExperimentConfig:
        """Create a minimal valid experiment config - use conftest helper with zero-defaults."""
        from tests.conftest import create_minimal_test_config
        # Use the conftest helper which has all required fields
        return create_minimal_test_config(
            metadata={
                'experiment_id': 'TEST',
                'scenario_id': 'TEST.1',
                'scenario_name': 'Test Scenario'
            },
            network={
                'num_banks': 20,
                'num_core_banks': 5
            },
            simulation={
                'num_runs': 10
            },
            output={
                'output_dir': 'results/test'
            }
        )

    def test_valid_config(self):
        """Test that valid config passes validation."""
        config = self.create_minimal_config()
        config.validate()  # Should not raise

    def test_bank_count_mismatch(self):
        """Test validation fails with bank count mismatch."""
        config = self.create_minimal_config()
        config.network.core_banks.count = 10  # Wrong count

        with pytest.raises(ValueError, match="Core banks count mismatch"):
            config.validate()

    def test_periphery_count_mismatch(self):
        """Test validation fails with periphery count mismatch."""
        config = self.create_minimal_config()
        config.network.periphery_banks.count = 10  # Wrong count

        with pytest.raises(ValueError, match="Periphery banks count mismatch"):
            config.validate()


# ============================================================================
# Test ConfigLoader
# ============================================================================

class TestConfigLoader:
    """Test ConfigLoader class."""

    @pytest.fixture
    def temp_config_file(self, tmp_path):
        """Create a temporary config file for testing."""
        config_data = {
            'metadata': {
                'experiment_id': 'E1',
                'scenario_id': 'E1.1',
                'scenario_name': 'Test Scenario',
                'hypothesis': 'Test hypothesis',
                'description': 'Test configuration',
                'tags': ['test']
            },
            'network': {
                'mode': 'stochastic',
                'topology': 'post_2008_reformed',
                'structure_seed': 42,
                'parameter_seed': 100,
                'num_banks': 20,
                'num_core_banks': 5,
                'core_banks': {
                    'capital_ratio': {
                        'distribution': 'uniform',
                        'min': 0.13,
                        'max': 0.17
                    },
                    'portfolio': {
                        'mortgage': {
                            'distribution': 'uniform',
                            'min': 0.15,
                            'max': 0.25
                        },
                        'remaining': {
                            'government_bond': 0.50,
                            'corporate_bond': 0.25,
                            'stock': 0.25
                        }
                    },
                    'total_assets': 500.0,
                    'external_assets_fraction': 0.92
                },
                'periphery_banks': {
                    'capital_ratio': {
                        'distribution': 'fixed',
                        'value': 0.12
                    },
                    'portfolio': {
                        'mortgage': {
                            'distribution': 'fixed',
                            'value': 0.25
                        },
                        'remaining': {
                            'government_bond': 0.35,
                            'corporate_bond': 0.30,
                            'stock': 0.35
                        }
                    },
                    'total_assets': 100.0,
                    'external_assets_fraction': 0.92
                },
                'connectivity': {
                    'core_to_core': 0.35,
                    'core_to_periphery': 0.15,
                    'periphery_to_core': 0.25,
                    'periphery_to_periphery': 0.10,
                    'core_to_core_exposure': {
                        'distribution': 'uniform',
                        'min': 2.0,
                        'max': 4.0
                    },
                    'core_to_periphery_exposure': {
                        'distribution': 'uniform',
                        'min': 1.0,
                        'max': 2.0
                    },
                    'periphery_to_core_exposure': {
                        'distribution': 'uniform',
                        'min': 0.5,
                        'max': 1.5
                    },
                    'periphery_to_periphery_exposure': {
                        'distribution': 'uniform',
                        'min': 0.3,
                        'max': 1.0
                    }
                }
            },
            'shock': {
                'mode': 'correlated',
                'asset_shocks': {
                    'mortgage': -0.20,
                    'corporate_bond': -0.10,
                    'government_bond': 0.02,
                    'stock': -0.15
                },
                'correlation': 0.90,
                'fire_sale_intensity': 0.15,
                'shock_volatility': 0.03
            },
            'simulation': {
                'num_runs': 200,
                'seed': 42,
                'fire_sales_enabled': True,
                'use_priority_claims': True
            },
            'output': {
                'output_dir': 'experiments/results/e1_correlation/corr_90',
                'save_summary': True,
                'save_detailed_results': False,
                'save_network_snapshots': False,
                'save_config_copy': False,
                'generate_plots': False,
                'plot_formats': ['png'],
                'verbose': False
            }
        }

        config_path = tmp_path / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        return config_path

    def test_load_config_file(self, temp_config_file):
        """Test loading a complete config file."""
        config = ConfigLoader.load(temp_config_file)

        # Check metadata
        assert config.metadata.experiment_id == 'E1'
        assert config.metadata.scenario_id == 'E1.1'
        assert config.metadata.scenario_name == 'Test Scenario'

        # Check network
        assert config.network.mode == 'stochastic'  # Mode stored as string
        assert config.network.topology == 'post_2008_reformed'
        assert config.network.num_banks == 20
        assert config.network.num_core_banks == 5

        # Check core banks
        assert isinstance(config.network.core_banks.capital_ratio, DistributionConfig)
        assert config.network.core_banks.capital_ratio.distribution == 'uniform'

        # Check shock
        assert config.shock.mode == 'correlated'  # Mode stored as string
        assert config.shock.correlation == 0.90
        assert config.shock.asset_shocks['mortgage'] == -0.20

        # Check simulation
        assert config.simulation.num_runs == 200
        assert config.simulation.seed == 42

        # Check output
        assert 'e1_correlation' in config.output.output_dir

    def test_load_config_file_not_found(self):
        """Test loading non-existent config file."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            ConfigLoader.load("nonexistent.yaml")

    def test_load_config_convenience_function(self, temp_config_file):
        """Test convenience function load_config."""
        config = load_config(temp_config_file)
        assert config.metadata.experiment_id == 'E1'

    def test_parse_distribution_in_config(self, temp_config_file):
        """Test parsing config with distribution values."""
        config = ConfigLoader.load(temp_config_file)

        # Check that distributions are parsed correctly
        assert isinstance(config.network.core_banks.capital_ratio, DistributionConfig)
        assert not config.network.core_banks.capital_ratio.distribution == 'fixed'
        assert config.network.core_banks.capital_ratio.min == 0.13
        assert config.network.core_banks.capital_ratio.max == 0.17


# ============================================================================
# Integration Tests
# ============================================================================

class TestConfigIntegration:
    """Integration tests for complete config workflow."""

    def test_full_workflow(self, tmp_path):
        """Test complete workflow: create YAML → load → validate."""
        # Create a complete config YAML
        config_data = {
            'metadata': {
                'experiment_id': 'E1',
                'scenario_id': 'E1.4',
                'scenario_name': 'High AI Correlation',
                'hypothesis': 'Correlation 0.90 increases failures to 70-85%',
                'description': 'Test scenario with high AI-driven correlation effects',
                'tags': ['correlation', 'AI']
            },
            'network': {
                'mode': 'stochastic',
                'topology': 'post_2008_reformed',
                'structure_seed': 42,
                'parameter_seed': 42,
                'num_banks': 20,
                'num_core_banks': 5,
                'core_banks': {
                    'capital_ratio': {
                        'distribution': 'uniform',
                        'min': 0.13,
                        'max': 0.17
                    },
                    'portfolio': {
                        'mortgage': {
                            'distribution': 'uniform',
                            'min': 0.15,
                            'max': 0.25
                        },
                        'remaining': {
                            'government_bond': 0.50,
                            'corporate_bond': 0.25,
                            'stock': 0.25
                        }
                    },
                    'total_assets': 500.0,
                    'external_assets_fraction': 0.9,
                    'count': 5
                },
                'periphery_banks': {
                    'capital_ratio': {
                        'distribution': 'uniform',
                        'min': 0.11,
                        'max': 0.14
                    },
                    'portfolio': {
                        'mortgage': {
                            'distribution': 'uniform',
                            'min': 0.20,
                            'max': 0.30
                        },
                        'remaining': {
                            'government_bond': 0.35,
                            'corporate_bond': 0.30,
                            'stock': 0.35
                        }
                    },
                    'total_assets': 100.0,
                    'external_assets_fraction': 0.92,
                    'count': 15
                },
                'connectivity': {
                    'core_to_core': 0.5,
                    'core_to_periphery': 0.4,
                    'periphery_to_core': 0.3,
                    'periphery_to_periphery': 0.2,
                    'core_to_core_exposure': {
                        'distribution': 'uniform',
                        'min': 3.0,
                        'max': 6.0
                    },
                    'core_to_periphery_exposure': {
                        'distribution': 'uniform',
                        'min': 1.0,
                        'max': 3.0
                    },
                    'periphery_to_core_exposure': {
                        'distribution': 'uniform',
                        'min': 0.5,
                        'max': 2.0
                    },
                    'periphery_to_periphery_exposure': {
                        'distribution': 'uniform',
                        'min': 0.3,
                        'max': 1.5
                    }
                }
            },
            'shock': {
                'mode': 'correlated',
                'asset_shocks': {
                    'mortgage': -0.20,
                    'corporate_bond': -0.10,
                    'government_bond': 0.02,
                    'stock': -0.15
                },
                'correlation': 0.90,
                'shock_volatility': 0.03,
                'fire_sale_intensity': 0.15
            },
            'simulation': {
                'num_runs': 200,
                'seed': 42,
                'fire_sales_enabled': True,
                'use_priority_claims': True
            },
            'output': {
                'output_dir': 'experiments/results/e1_correlation/corr_90',
                'save_summary': True,
                'save_detailed_results': True,
                'save_network_snapshots': False,
                'save_config_copy': True,
                'generate_plots': True,
                'plot_formats': ['png'],
                'verbose': True
            }
        }

        # Write config file
        config_path = tmp_path / "integration_test.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        # Load config
        config = load_config(config_path)

        # Validate
        config.validate()

        # Verify all sections loaded correctly
        assert config.metadata.experiment_id == 'E1'
        assert config.network.mode == 'stochastic'  # Mode stored as string, not enum
        assert config.shock.mode == 'correlated'  # Mode stored as string, not enum
        assert config.simulation.num_runs == 200
        assert 'e1_correlation' in config.output.output_dir

        # Test sampling from distributions
        rng = np.random.RandomState(42)
        if isinstance(config.network.core_banks.capital_ratio, DistributionConfig):
            capital = config.network.core_banks.capital_ratio.sample(rng)
            assert 0.13 <= capital <= 0.17


# ============================================================================
# Test Bank Subgroups (for realistic heterogeneity)
# ============================================================================

class TestBankSubgroups:
    """Test bank subgroup functionality for heterogeneous bank types."""

    @pytest.fixture
    def config_with_subgroups(self, tmp_path):
        """Create a config file with bank subgroups."""
        config_data = {
            'metadata': {
                'experiment_id': 'TEST_SUBGROUPS',
                'scenario_id': 'TEST.1',
                'scenario_name': 'Test Subgroups',
                'hypothesis': 'Test subgroup functionality',
                'description': 'Test configuration for bank subgroups',
                'tags': ['test', 'subgroups']
            },
            'network': {
                'mode': 'stochastic',
                'topology': 'heterogeneous',
                'num_banks': 10,
                'num_core_banks': 3,
                'structure_seed': 42,
                'parameter_seed': 100,
                'core_bank_subgroups': [
                    {
                        'count': 2,
                        'capital_ratio': {
                            'distribution': 'fixed',
                            'value': 0.15
                        },
                        'portfolio': {
                            'mortgage': {
                                'distribution': 'fixed',
                                'value': 0.2
                            },
                            'remaining': {
                                'government_bond': 0.5,    # 0.4 / 0.8
                                'corporate_bond': 0.375,   # 0.3 / 0.8
                                'stock': 0.125             # 0.1 / 0.8
                            }
                        },
                        'total_assets': 500.0,
                        'external_assets_fraction': 0.85
                    },
                    {
                        'count': 1,
                        'capital_ratio': {
                            'distribution': 'fixed',
                            'value': 0.10
                        },
                        'portfolio': {
                            'mortgage': {
                                'distribution': 'fixed',
                                'value': 0.5
                            },
                            'remaining': {
                                'government_bond': 0.4,   # 0.2 / 0.5
                                'corporate_bond': 0.4,    # 0.2 / 0.5
                                'stock': 0.2              # 0.1 / 0.5
                            }
                        },
                        'total_assets': 300.0,
                        'external_assets_fraction': 0.85
                    }
                ],
                'periphery_bank_subgroups': [
                    {
                        'count': 5,
                        'capital_ratio': {
                            'distribution': 'fixed',
                            'value': 0.12
                        },
                        'portfolio': {
                            'mortgage': {
                                'distribution': 'fixed',
                                'value': 0.4
                            },
                            'remaining': {
                                'government_bond': 0.5,   # 0.3 / 0.6
                                'corporate_bond': 0.333,  # 0.2 / 0.6
                                'stock': 0.167            # 0.1 / 0.6
                            }
                        },
                        'total_assets': 100.0,
                        'external_assets_fraction': 0.90
                    },
                    {
                        'count': 2,
                        'capital_ratio': {
                            'distribution': 'fixed',
                            'value': 0.14
                        },
                        'portfolio': {
                            'mortgage': {
                                'distribution': 'fixed',
                                'value': 0.2
                            },
                            'remaining': {
                                'government_bond': 0.625,  # 0.5 / 0.8
                                'corporate_bond': 0.25,    # 0.2 / 0.8
                                'stock': 0.125             # 0.1 / 0.8
                            }
                        },
                        'total_assets': 80.0,
                        'external_assets_fraction': 0.90
                    }
                ],
                'connectivity': {
                    'core_to_core': 0.35,
                    'core_to_periphery': 0.15,
                    'periphery_to_core': 0.10,
                    'periphery_to_periphery': 0.05,
                    'core_to_core_exposure': {
                        'distribution': 'fixed',
                        'value': 5.0
                    },
                    'core_to_periphery_exposure': {
                        'distribution': 'fixed',
                        'value': 2.0
                    },
                    'periphery_to_core_exposure': {
                        'distribution': 'fixed',
                        'value': 1.0
                    },
                    'periphery_to_periphery_exposure': {
                        'distribution': 'fixed',
                        'value': 0.5
                    }
                }
            },
            'shock': {
                'mode': 'correlated',
                'asset_shocks': {
                    'mortgage': -0.20,
                    'corporate_bond': -0.10,
                    'stock': -0.15,
                    'government_bond': 0.0
                },
                'correlation': 0.60,
                'fire_sale_intensity': 0.15,
                'shock_volatility': 0.0
            },
            'simulation': {
                'num_runs': 100,
                'seed': 42,
                'fire_sales_enabled': True,
                'use_priority_claims': True
            },
            'output': {
                'output_dir': 'test_subgroups',
                'save_summary': True,
                'save_detailed_results': False,
                'save_network_snapshots': False,
                'save_config_copy': False,
                'generate_plots': False,
                'plot_formats': ['png'],
                'verbose': False
            }
        }

        config_path = tmp_path / "subgroups_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        return config_path

    def test_load_config_with_subgroups(self, config_with_subgroups):
        """Test loading a config file with bank subgroups."""
        config = ConfigLoader.load(config_with_subgroups)

        # Check that subgroups were loaded
        assert config.network.core_bank_subgroups is not None
        assert config.network.periphery_bank_subgroups is not None
        assert len(config.network.core_bank_subgroups) == 2
        assert len(config.network.periphery_bank_subgroups) == 2

        # Check first core subgroup
        core_subgroup_1 = config.network.core_bank_subgroups[0]
        assert core_subgroup_1.count == 2
        assert core_subgroup_1.capital_ratio.value == 0.15
        assert core_subgroup_1.portfolio.mortgage.value == 0.2

        # Check second core subgroup
        core_subgroup_2 = config.network.core_bank_subgroups[1]
        assert core_subgroup_2.count == 1
        assert core_subgroup_2.capital_ratio.value == 0.10
        assert core_subgroup_2.portfolio.mortgage.value == 0.5

    def test_get_core_bank_groups_with_subgroups(self, config_with_subgroups):
        """Test get_core_bank_groups() returns subgroups when available."""
        config = ConfigLoader.load(config_with_subgroups)
        groups = config.network.get_core_bank_groups()

        assert len(groups) == 2
        assert groups[0].count == 2
        assert groups[1].count == 1

    def test_get_periphery_bank_groups_with_subgroups(self, config_with_subgroups):
        """Test get_periphery_bank_groups() returns subgroups when available."""
        config = ConfigLoader.load(config_with_subgroups)
        groups = config.network.get_periphery_bank_groups()

        assert len(groups) == 2
        assert groups[0].count == 5
        assert groups[1].count == 2

    def test_has_subgroups(self, config_with_subgroups):
        """Test has_subgroups() returns True when subgroups are present."""
        config = ConfigLoader.load(config_with_subgroups)
        assert config.network.has_subgroups() is True

    def test_validate_subgroup_counts(self, config_with_subgroups):
        """Test validation of subgroup counts."""
        config = ConfigLoader.load(config_with_subgroups)

        # Should not raise - counts match (2+1=3 core, 5+2=7 periphery)
        config.validate()

    def test_validate_subgroup_counts_mismatch(self, tmp_path):
        """Test validation fails when subgroup counts don't match."""
        config_data = {
            'metadata': {
                'experiment_id': 'TEST',
                'scenario_id': 'TEST.1',
                'scenario_name': 'Test',
                'hypothesis': 'Test count mismatch',
                'description': 'Test configuration with mismatched counts',
                'tags': ['test']
            },
            'network': {
                'mode': 'stochastic',
                'topology': 'test',
                'num_banks': 10,
                'num_core_banks': 5,  # Says 5 core banks
                'structure_seed': 42,
                'parameter_seed': 100,
                'core_bank_subgroups': [
                    {
                        'count': 2,  # But only 2+1=3 in subgroups!
                        'capital_ratio': {
                            'distribution': 'fixed',
                            'value': 0.15
                        },
                        'portfolio': {
                            'mortgage': {
                                'distribution': 'fixed',
                                'value': 0.2
                            },
                            'remaining': {
                                'government_bond': 0.5,
                                'corporate_bond': 0.375,
                                'stock': 0.125
                            }
                        },
                        'total_assets': 500.0,
                        'external_assets_fraction': 0.85
                    },
                    {
                        'count': 1,
                        'capital_ratio': {
                            'distribution': 'fixed',
                            'value': 0.10
                        },
                        'portfolio': {
                            'mortgage': {
                                'distribution': 'fixed',
                                'value': 0.2
                            },
                            'remaining': {
                                'government_bond': 0.5,
                                'corporate_bond': 0.375,
                                'stock': 0.125
                            }
                        },
                        'total_assets': 300.0,
                        'external_assets_fraction': 0.85
                    }
                ],
                'periphery_bank_subgroups': [
                    {
                        'count': 5,
                        'capital_ratio': {
                            'distribution': 'fixed',
                            'value': 0.12
                        },
                        'portfolio': {
                            'mortgage': {
                                'distribution': 'fixed',
                                'value': 0.2
                            },
                            'remaining': {
                                'government_bond': 0.5,
                                'corporate_bond': 0.375,
                                'stock': 0.125
                            }
                        },
                        'total_assets': 100.0,
                        'external_assets_fraction': 0.90
                    }
                ],
                'connectivity': {
                    'core_to_core': 0.35,
                    'core_to_periphery': 0.15,
                    'periphery_to_core': 0.10,
                    'periphery_to_periphery': 0.05,
                    'core_to_core_exposure': {
                        'distribution': 'fixed',
                        'value': 5.0
                    },
                    'core_to_periphery_exposure': {
                        'distribution': 'fixed',
                        'value': 2.0
                    },
                    'periphery_to_core_exposure': {
                        'distribution': 'fixed',
                        'value': 1.0
                    },
                    'periphery_to_periphery_exposure': {
                        'distribution': 'fixed',
                        'value': 0.5
                    }
                }
            },
            'shock': {
                'mode': 'correlated',
                'asset_shocks': {
                    'mortgage': -0.20,
                    'government_bond': 0.0,
                    'corporate_bond': 0.0,
                    'stock': 0.0
                },
                'correlation': 0.60,
                'fire_sale_intensity': 0.15,
                'shock_volatility': 0.0
            },
            'simulation': {
                'num_runs': 100,
                'seed': 42,
                'fire_sales_enabled': True,
                'use_priority_claims': True
            },
            'output': {
                'output_dir': 'test',
                'save_summary': True,
                'save_detailed_results': False,
                'save_network_snapshots': False,
                'save_config_copy': False,
                'generate_plots': False,
                'plot_formats': ['png'],
                'verbose': False
            }
        }

        config_path = tmp_path / "invalid_counts.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        # Should raise validation error during load
        with pytest.raises(ValueError, match="Core banks count mismatch"):
            config = ConfigLoader.load(config_path)

    def test_backward_compatibility_single_group(self, tmp_path):
        """Test that old configs without subgroups still work."""
        # Create an old-style config without subgroups
        config_data = {
            'metadata': {
                'experiment_id': 'TEST',
                'scenario_id': 'TEST.1',
                'scenario_name': 'Test Backward Compatibility',
                'hypothesis': 'Test backward compatibility',
                'description': 'Test configuration for backward compatibility',
                'tags': ['test', 'backward-compatibility']
            },
            'network': {
                'mode': 'stochastic',
                'topology': 'test',
                'num_banks': 10,
                'num_core_banks': 3,
                'structure_seed': 42,
                'parameter_seed': 100,
                # Old style - single group (no subgroups)
                'core_banks': {
                    'capital_ratio': {
                        'distribution': 'fixed',
                        'value': 0.15
                    },
                    'portfolio': {
                        'mortgage': {
                            'distribution': 'fixed',
                            'value': 0.2
                        },
                        'remaining': {
                            'government_bond': 0.5,
                            'corporate_bond': 0.375,
                            'stock': 0.125
                        }
                    },
                    'total_assets': 500.0,
                    'external_assets_fraction': 0.85
                },
                'periphery_banks': {
                    'capital_ratio': {
                        'distribution': 'fixed',
                        'value': 0.12
                    },
                    'portfolio': {
                        'mortgage': {
                            'distribution': 'fixed',
                            'value': 0.3
                        },
                        'remaining': {
                            'government_bond': 0.429,
                            'corporate_bond': 0.429,
                            'stock': 0.142
                        }
                    },
                    'total_assets': 100.0,
                    'external_assets_fraction': 0.90
                },
                'connectivity': {
                    'core_to_core': 0.35,
                    'core_to_periphery': 0.15,
                    'periphery_to_core': 0.10,
                    'periphery_to_periphery': 0.05,
                    'core_to_core_exposure': {
                        'distribution': 'fixed',
                        'value': 5.0
                    },
                    'core_to_periphery_exposure': {
                        'distribution': 'fixed',
                        'value': 2.0
                    },
                    'periphery_to_core_exposure': {
                        'distribution': 'fixed',
                        'value': 1.0
                    },
                    'periphery_to_periphery_exposure': {
                        'distribution': 'fixed',
                        'value': 0.5
                    }
                }
            },
            'shock': {
                'mode': 'correlated',
                'asset_shocks': {
                    'mortgage': -0.20,
                    'government_bond': 0.0,
                    'corporate_bond': 0.0,
                    'stock': 0.0
                },
                'correlation': 0.60,
                'fire_sale_intensity': 0.15,
                'shock_volatility': 0.0
            },
            'simulation': {
                'num_runs': 100,
                'seed': 42,
                'fire_sales_enabled': True,
                'use_priority_claims': True
            },
            'output': {
                'output_dir': 'test',
                'save_summary': True,
                'save_detailed_results': False,
                'save_network_snapshots': False,
                'save_config_copy': False,
                'generate_plots': False,
                'plot_formats': ['png'],
                'verbose': False
            }
        }

        config_path = tmp_path / "old_style_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        config = ConfigLoader.load(config_path)

        # Should not have subgroups
        assert config.network.core_bank_subgroups is None
        assert config.network.periphery_bank_subgroups is None
        assert config.network.has_subgroups() is False

        # But should still work with helper methods
        core_groups = config.network.get_core_bank_groups()
        assert len(core_groups) == 1
        assert core_groups[0] == config.network.core_banks

        periphery_groups = config.network.get_periphery_bank_groups()
        assert len(periphery_groups) == 1
        assert periphery_groups[0] == config.network.periphery_banks

    def test_cannot_specify_both_single_and_subgroups(self, tmp_path):
        """Test that specifying both single group and subgroups raises error."""
        config_data = {
            'metadata': {
                'experiment_id': 'TEST',
                'scenario_id': 'TEST.1',
                'scenario_name': 'Test',
                'hypothesis': 'Test precedence',
                'description': 'Test configuration for subgroup precedence',
                'tags': ['test']
            },
            'network': {
                'mode': 'stochastic',
                'topology': 'test',
                'num_banks': 5,
                'num_core_banks': 2,
                'structure_seed': 42,
                'parameter_seed': 100,
                # Both specified - subgroups should win
                'core_banks': {
                    'capital_ratio': {
                        'distribution': 'fixed',
                        'value': 0.15
                    },
                    'portfolio': {
                        'mortgage': {
                            'distribution': 'fixed',
                            'value': 0.2
                        },
                        'remaining': {
                            'government_bond': 0.5,
                            'corporate_bond': 0.375,
                            'stock': 0.125
                        }
                    },
                    'total_assets': 500.0,
                    'external_assets_fraction': 0.85
                },
                'core_bank_subgroups': [
                    {
                        'count': 2,
                        'capital_ratio': {
                            'distribution': 'fixed',
                            'value': 0.20  # Different value
                        },
                        'portfolio': {
                            'mortgage': {
                                'distribution': 'fixed',
                                'value': 0.2
                            },
                            'remaining': {
                                'government_bond': 0.5,
                                'corporate_bond': 0.375,
                                'stock': 0.125
                            }
                        },
                        'total_assets': 600.0,
                        'external_assets_fraction': 0.85
                    }
                ],
                'periphery_banks': {
                    'capital_ratio': {
                        'distribution': 'fixed',
                        'value': 0.12
                    },
                    'portfolio': {
                        'mortgage': {
                            'distribution': 'fixed',
                            'value': 0.2
                        },
                        'remaining': {
                            'government_bond': 0.5,
                            'corporate_bond': 0.375,
                            'stock': 0.125
                        }
                    },
                    'total_assets': 100.0,
                    'external_assets_fraction': 0.90
                },
                'connectivity': {
                    'core_to_core': 0.35,
                    'core_to_periphery': 0.15,
                    'periphery_to_core': 0.10,
                    'periphery_to_periphery': 0.05,
                    'core_to_core_exposure': {
                        'distribution': 'fixed',
                        'value': 5.0
                    },
                    'core_to_periphery_exposure': {
                        'distribution': 'fixed',
                        'value': 2.0
                    },
                    'periphery_to_core_exposure': {
                        'distribution': 'fixed',
                        'value': 1.0
                    },
                    'periphery_to_periphery_exposure': {
                        'distribution': 'fixed',
                        'value': 0.5
                    }
                }
            },
            'shock': {
                'mode': 'correlated',
                'asset_shocks': {
                    'mortgage': -0.20,
                    'government_bond': 0.0,
                    'corporate_bond': 0.0,
                    'stock': 0.0
                },
                'correlation': 0.60,
                'fire_sale_intensity': 0.15,
                'shock_volatility': 0.0
            },
            'simulation': {
                'num_runs': 100,
                'seed': 42,
                'fire_sales_enabled': True,
                'use_priority_claims': True
            },
            'output': {
                'output_dir': 'test',
                'save_summary': True,
                'save_detailed_results': False,
                'save_network_snapshots': False,
                'save_config_copy': False,
                'generate_plots': False,
                'plot_formats': ['png'],
                'verbose': False
            }
        }

        config_path = tmp_path / "both_specified.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        # Should raise validation error - cannot specify both
        with pytest.raises(ValueError, match="Cannot specify both core_banks and core_bank_subgroups"):
            config = ConfigLoader.load(config_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
