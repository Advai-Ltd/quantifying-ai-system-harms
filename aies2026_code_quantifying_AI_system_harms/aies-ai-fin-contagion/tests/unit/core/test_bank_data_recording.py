"""
Tests for comprehensive bank data recording.

Ensures all critical bank data is captured in to_dict() for validation
and debugging of Options A, D, and future modifications.
"""

import pytest
from financial_contagion_networks.core.bank import Bank


class TestBankDataRecording:
    """Test that to_dict() captures all essential bank data."""

    def test_interbank_assets_fraction_recorded(self):
        """Test that interbank_assets_fraction is calculated and recorded."""
        bank = Bank(
            bank_id=0,
            external_assets=850.0,
            interbank_assets=150.0,  # 15% of total assets
            interbank_liabilities=100.0,
            external_liabilities=900.0
        )

        data = bank.to_dict()

        assert 'interbank_assets_fraction' in data
        assert data['interbank_assets_fraction'] == pytest.approx(0.15, abs=1e-6)

    def test_interbank_assets_fraction_various_values(self):
        """Test interbank_assets_fraction with various values."""
        test_cases = [
            # (external_assets, interbank_assets, expected_fraction)
            (900.0, 100.0, 0.10),  # 10%
            (850.0, 150.0, 0.15),  # 15%
            (700.0, 300.0, 0.30),  # 30%
            (950.0, 50.0, 0.05),   # 5%
            (920.0, 80.0, 0.08),   # 8%
        ]

        for external, interbank, expected_fraction in test_cases:
            bank = Bank(
                bank_id=0,
                external_assets=external,
                interbank_assets=interbank,
                interbank_liabilities=100.0,
                external_liabilities=900.0
            )

            data = bank.to_dict()
            assert data['interbank_assets_fraction'] == pytest.approx(expected_fraction, abs=1e-6), \
                f"Failed for external={external}, interbank={interbank}"

    def test_interbank_assets_fraction_no_interbank(self):
        """Test that interbank_assets_fraction is 0 when no interbank assets."""
        bank = Bank(
            bank_id=0,
            external_assets=1000.0,
            interbank_assets=0.0,  # No interbank assets
            interbank_liabilities=0.0,
            external_liabilities=900.0
        )

        data = bank.to_dict()

        assert 'interbank_assets_fraction' in data
        assert data['interbank_assets_fraction'] == 0.0

    def test_all_critical_fields_present(self):
        """Test that to_dict() contains all critical fields for analysis."""
        bank = Bank(
            bank_id=5,
            external_assets=850.0,
            interbank_assets=150.0,
            interbank_liabilities=100.0,
            external_liabilities=900.0
        )

        data = bank.to_dict()

        # Critical fields for validation
        required_fields = [
            'bank_id',
            'external_assets',
            'interbank_assets',
            'interbank_liabilities',
            'external_liabilities',
            'total_assets',
            'total_liabilities',
            'equity',
            'capital_ratio',
            'status',
            'recovery_rate',
            'loss_tracking',
            'failure_metadata',
            'interbank_assets_fraction',  # NEW - Option A validation
        ]

        for field in required_fields:
            assert field in data, f"Missing critical field: {field}"

    def test_interbank_assets_fraction_after_loss(self):
        """Test that interbank_assets_fraction updates correctly after losses."""
        bank = Bank(
            bank_id=0,
            external_assets=850.0,
            interbank_assets=150.0,
            interbank_liabilities=100.0,
            external_liabilities=900.0
        )

        # Initial fraction: 150 / 1000 = 15%
        initial_data = bank.to_dict()
        assert initial_data['interbank_assets_fraction'] == pytest.approx(0.15, abs=1e-6)

        # Apply loss to external assets
        bank.external_assets -= 100.0  # Reduce external by 100

        # New fraction: 150 / 900 = 16.67% (interbank assets unchanged but total decreased)
        after_loss_data = bank.to_dict()
        assert after_loss_data['interbank_assets_fraction'] == pytest.approx(0.1667, abs=1e-3)

    def test_to_dict_json_serializable(self):
        """Test that to_dict() output is JSON serializable."""
        import json

        bank = Bank(
            bank_id=0,
            external_assets=850.0,
            interbank_assets=150.0,
            interbank_liabilities=100.0,
            external_liabilities=900.0
        )

        data = bank.to_dict()

        # Should not raise exception
        json_str = json.dumps(data)
        assert json_str is not None

        # Should round-trip correctly
        parsed = json.loads(json_str)
        assert parsed['bank_id'] == 0
        assert parsed['interbank_assets_fraction'] == pytest.approx(0.15, abs=1e-6)
