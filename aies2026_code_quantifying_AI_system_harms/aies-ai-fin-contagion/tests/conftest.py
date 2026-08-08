"""
Shared test fixtures and utilities for the test suite.

This module provides helper functions for creating valid test configs
that comply with the zero-defaults Pydantic schema.
"""

import pytest
from financial_contagion_networks.config import ExperimentConfig


def create_minimal_test_config(**overrides):
    """
    Create a complete valid test config with all required fields.

    This helper ensures all tests use configs that comply with the
    zero-defaults policy. All fields are explicitly specified.

    Args:
        **overrides: Dict overrides for any config section
                    Supports deep merging for nested updates

    Returns:
        ExperimentConfig with all required fields populated

    Example:
        # Create config with custom simulation params
        config = create_minimal_test_config(
            simulation={'num_runs': 100, 'seed': 123}
        )

        # Create config with custom network
        config = create_minimal_test_config(
            network={'num_banks': 20, 'num_core_banks': 5}
        )
    """
    base_config = {
        'metadata': {
            'experiment_id': 'TEST',
            'scenario_id': 'TEST_SCENARIO',
            'scenario_name': 'Test Scenario',
            'hypothesis': 'Test hypothesis',
            'description': 'Test description',
            'tags': ['test']
        },
        'network': {
            'mode': 'stochastic',
            'topology': 'test',
            'num_banks': 10,
            'num_core_banks': 3,
            'structure_seed': 42,
            'parameter_seed': 100,
            'core_banks': {
                'capital_ratio': {'distribution': 'fixed', 'value': 0.15},
                'portfolio': {
                    'mortgage': {'distribution': 'fixed', 'value': 0.2},
                    'remaining': {
                        'government_bond': 0.35,
                        'corporate_bond': 0.40,
                        'stock': 0.25
                    }
                },
                'total_assets': 500.0,
                'external_assets_fraction': 0.9
            },
            'periphery_banks': {
                'capital_ratio': {'distribution': 'fixed', 'value': 0.12},
                'portfolio': {
                    'mortgage': {'distribution': 'fixed', 'value': 0.25},
                    'remaining': {
                        'government_bond': 0.30,
                        'corporate_bond': 0.35,
                        'stock': 0.35
                    }
                },
                'total_assets': 100.0,
                'external_assets_fraction': 0.92
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
                'government_bond': 0.02,
                'corporate_bond': -0.1,
                'mortgage': -0.2,
                'stock': -0.15
            },
            'correlation': 0.6,
            'fire_sale_intensity': 0.15,
            'shock_volatility': 0.03
        },
        'simulation': {
            'num_runs': 10,
            'seed': 42,
            'fire_sales_enabled': True,
            'use_priority_claims': False
        },
        'output': {
            'output_dir': '/tmp/test_output',
            'save_summary': True,
            'save_detailed_results': False,
            'save_network_snapshots': False,
            'save_config_copy': False,
            'generate_plots': False,
            'plot_formats': ['png'],
            'verbose': False
        }
    }

    # Apply overrides with deep merge
    def deep_merge(base, overrides):
        """Recursively merge overrides into base dict."""
        for key, value in overrides.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_merge(base[key], value)
            else:
                base[key] = value

    if overrides:
        deep_merge(base_config, overrides)

    return ExperimentConfig(**base_config)


@pytest.fixture
def minimal_test_config():
    """
    Pytest fixture providing a complete minimal test config.

    Usage:
        def test_something(minimal_test_config):
            config = minimal_test_config
            # Use config in test
    """
    return create_minimal_test_config()


@pytest.fixture
def test_config_dict():
    """
    Pytest fixture providing a test config as a dict (before Pydantic validation).

    Useful for testing config loading and validation.

    Usage:
        def test_config_loading(test_config_dict):
            # Modify dict as needed
            test_config_dict['simulation']['num_runs'] = 100
            # Load with Pydantic
            config = ExperimentConfig(**test_config_dict)
    """
    return create_minimal_test_config().model_dump()
