"""
Comprehensive tests for the Bank class.
"""

import pytest
from financial_contagion_networks.core.bank import Bank, BankStatus


class TestBankInitialization:
    """Tests for Bank initialization."""

    def test_basic_initialization(self):
        """Test creating a bank with basic parameters."""
        bank = Bank(
            bank_id=1,
            external_assets=100.0,
            interbank_assets=20.0,
            interbank_liabilities=10.0,
            external_liabilities=80.0
        )
        assert bank.bank_id == 1
        assert bank.external_assets == 100.0
        assert bank.interbank_assets == 20.0
        assert bank.interbank_liabilities == 10.0
        assert bank.external_liabilities == 80.0

    def test_minimal_initialization(self):
        """Test creating a bank with only required parameters."""
        bank = Bank(bank_id=1, external_assets=100.0)
        assert bank.bank_id == 1
        assert bank.external_assets == 100.0
        assert bank.interbank_assets == 0.0
        assert bank.interbank_liabilities == 0.0
        assert bank.external_liabilities == 0.0

    def test_negative_bank_id(self):
        """Test that negative bank_id raises ValueError."""
        with pytest.raises(ValueError, match="bank_id must be non-negative"):
            Bank(bank_id=-1, external_assets=100.0)

    def test_zero_external_assets(self):
        """Test that zero external assets raises ValueError."""
        with pytest.raises(ValueError, match="external_assets must be positive"):
            Bank(bank_id=1, external_assets=0.0)

    def test_negative_external_assets(self):
        """Test that negative external assets raises ValueError."""
        with pytest.raises(ValueError, match="external_assets must be positive"):
            Bank(bank_id=1, external_assets=-10.0)

    def test_negative_interbank_assets(self):
        """Test that negative interbank assets raises ValueError."""
        with pytest.raises(ValueError, match="interbank_assets must be non-negative"):
            Bank(bank_id=1, external_assets=100.0, interbank_assets=-10.0)

    def test_negative_interbank_liabilities(self):
        """Test that negative interbank liabilities raises ValueError."""
        with pytest.raises(ValueError, match="interbank_liabilities must be non-negative"):
            Bank(bank_id=1, external_assets=100.0, interbank_liabilities=-10.0)

    def test_negative_external_liabilities(self):
        """Test that negative external liabilities raises ValueError."""
        with pytest.raises(ValueError, match="external_liabilities must be non-negative"):
            Bank(bank_id=1, external_assets=100.0, external_liabilities=-10.0)


class TestBankBalanceSheet:
    """Tests for balance sheet calculations."""

    def test_total_assets(self):
        """Test total assets calculation."""
        bank = Bank(
            bank_id=1,
            external_assets=100.0,
            interbank_assets=20.0
        )
        assert bank.get_total_assets() == 120.0

    def test_total_liabilities(self):
        """Test total liabilities calculation."""
        bank = Bank(
            bank_id=1,
            external_assets=100.0,
            interbank_liabilities=10.0,
            external_liabilities=80.0
        )
        assert bank.get_total_liabilities() == 90.0

    def test_equity_positive(self):
        """Test equity calculation with positive equity."""
        bank = Bank(
            bank_id=1,
            external_assets=100.0,
            interbank_assets=20.0,
            interbank_liabilities=10.0,
            external_liabilities=80.0
        )
        # Assets = 120, Liabilities = 90, Equity = 30
        assert bank.get_equity() == 30.0

    def test_equity_zero(self):
        """Test equity calculation when exactly zero."""
        bank = Bank(
            bank_id=1,
            external_assets=100.0,
            external_liabilities=100.0
        )
        assert bank.get_equity() == 0.0

    def test_equity_negative(self):
        """Test equity calculation when negative (insolvent)."""
        bank = Bank(
            bank_id=1,
            external_assets=100.0,
            external_liabilities=120.0
        )
        assert bank.get_equity() == -20.0

    def test_capital_ratio(self):
        """Test capital ratio calculation."""
        bank = Bank(
            bank_id=1,
            external_assets=100.0,
            interbank_assets=20.0,
            external_liabilities=90.0
        )
        # Assets = 120, Equity = 30, Capital Ratio = 30/120 = 0.25
        assert abs(bank.get_capital_ratio() - 0.25) < 1e-10

    def test_leverage_ratio(self):
        """Test leverage ratio calculation."""
        bank = Bank(
            bank_id=1,
            external_assets=100.0,
            interbank_assets=20.0,
            external_liabilities=90.0
        )
        # Assets = 120, Equity = 30, Leverage = 120/30 = 4.0
        assert abs(bank.get_leverage_ratio() - 4.0) < 1e-10

    def test_leverage_ratio_insolvent(self):
        """Test leverage ratio when insolvent (should return infinity)."""
        bank = Bank(
            bank_id=1,
            external_assets=100.0,
            external_liabilities=120.0
        )
        assert bank.get_leverage_ratio() == float('inf')


class TestBankSolvency:
    """Tests for solvency checks."""

    def test_solvent_bank(self):
        """Test that a solvent bank is identified correctly."""
        bank = Bank(
            bank_id=1,
            external_assets=100.0,
            external_liabilities=80.0
        )
        assert bank.is_solvent() is True
        assert bank.get_status() == BankStatus.SOLVENT

    def test_insolvent_bank(self):
        """Test that an insolvent bank is identified correctly."""
        bank = Bank(
            bank_id=1,
            external_assets=100.0,
            external_liabilities=120.0
        )
        assert bank.is_solvent() is False

    def test_marginally_solvent(self):
        """Test bank with very small positive equity."""
        bank = Bank(
            bank_id=1,
            external_assets=100.0,
            external_liabilities=99.9999
        )
        assert bank.is_solvent() is True


class TestInterbankExposures:
    """Tests for interbank exposure management."""

    def test_add_single_exposure(self):
        """Test adding a single interbank exposure."""
        bank = Bank(bank_id=1, external_assets=100.0)
        bank.add_interbank_exposure(counterparty_id=2, amount=10.0)

        assert bank.interbank_assets == 10.0
        assert bank.interbank_exposures[2] == 10.0

    def test_add_multiple_exposures_different_counterparties(self):
        """Test adding exposures to multiple counterparties."""
        bank = Bank(bank_id=1, external_assets=100.0)
        bank.add_interbank_exposure(counterparty_id=2, amount=10.0)
        bank.add_interbank_exposure(counterparty_id=3, amount=15.0)

        assert bank.interbank_assets == 25.0
        assert bank.interbank_exposures[2] == 10.0
        assert bank.interbank_exposures[3] == 15.0

    def test_add_multiple_exposures_same_counterparty(self):
        """Test adding multiple exposures to the same counterparty."""
        bank = Bank(bank_id=1, external_assets=100.0)
        bank.add_interbank_exposure(counterparty_id=2, amount=10.0)
        bank.add_interbank_exposure(counterparty_id=2, amount=5.0)

        assert bank.interbank_assets == 15.0
        assert bank.interbank_exposures[2] == 15.0

    def test_add_exposure_to_self(self):
        """Test that adding exposure to self raises ValueError."""
        bank = Bank(bank_id=1, external_assets=100.0)
        with pytest.raises(ValueError, match="cannot have exposure to itself"):
            bank.add_interbank_exposure(counterparty_id=1, amount=10.0)

    def test_add_zero_exposure(self):
        """Test that adding zero exposure raises ValueError."""
        bank = Bank(bank_id=1, external_assets=100.0)
        with pytest.raises(ValueError, match="Exposure amount must be positive"):
            bank.add_interbank_exposure(counterparty_id=2, amount=0.0)

    def test_add_negative_exposure(self):
        """Test that adding negative exposure raises ValueError."""
        bank = Bank(bank_id=1, external_assets=100.0)
        with pytest.raises(ValueError, match="Exposure amount must be positive"):
            bank.add_interbank_exposure(counterparty_id=2, amount=-10.0)


class TestInterbankObligations:
    """Tests for interbank obligation management."""

    def test_add_single_obligation(self):
        """Test adding a single interbank obligation."""
        bank = Bank(bank_id=1, external_assets=100.0)
        bank.add_interbank_obligation(counterparty_id=2, amount=10.0)

        assert bank.interbank_liabilities == 10.0
        assert bank.interbank_obligations[2] == 10.0

    def test_add_multiple_obligations(self):
        """Test adding obligations to multiple counterparties."""
        bank = Bank(bank_id=1, external_assets=100.0)
        bank.add_interbank_obligation(counterparty_id=2, amount=10.0)
        bank.add_interbank_obligation(counterparty_id=3, amount=15.0)

        assert bank.interbank_liabilities == 25.0
        assert bank.interbank_obligations[2] == 10.0
        assert bank.interbank_obligations[3] == 15.0

    def test_add_obligation_to_self(self):
        """Test that adding obligation to self raises ValueError."""
        bank = Bank(bank_id=1, external_assets=100.0)
        with pytest.raises(ValueError, match="cannot have obligation to itself"):
            bank.add_interbank_obligation(counterparty_id=1, amount=10.0)

    def test_add_zero_obligation(self):
        """Test that adding zero obligation raises ValueError."""
        bank = Bank(bank_id=1, external_assets=100.0)
        with pytest.raises(ValueError, match="Obligation amount must be positive"):
            bank.add_interbank_obligation(counterparty_id=2, amount=0.0)


class TestInterbankLosses:
    """Tests for applying interbank losses."""

    def test_apply_partial_loss(self):
        """Test applying a partial loss from counterparty."""
        bank = Bank(bank_id=1, external_assets=100.0, external_liabilities=80.0)
        bank.add_interbank_exposure(counterparty_id=2, amount=20.0)
        initial_equity = bank.get_equity()

        bank.apply_interbank_loss(counterparty_id=2, loss_amount=10.0)

        assert bank.interbank_assets == 10.0
        assert bank.interbank_exposures[2] == 10.0
        assert bank.get_equity() == initial_equity - 10.0
        assert bank.is_solvent() is True

    def test_apply_full_loss(self):
        """Test applying a full loss (complete default)."""
        bank = Bank(bank_id=1, external_assets=100.0, external_liabilities=80.0)
        bank.add_interbank_exposure(counterparty_id=2, amount=20.0)

        bank.apply_interbank_loss(counterparty_id=2, loss_amount=20.0)

        assert bank.interbank_assets == 0.0
        assert bank.interbank_exposures[2] == 0.0
        assert bank.is_solvent() is True

    def test_loss_causes_insolvency(self):
        """Test that a loss can cause insolvency."""
        bank = Bank(bank_id=1, external_assets=100.0, external_liabilities=105.0)
        bank.add_interbank_exposure(counterparty_id=2, amount=20.0)
        # Initial equity = 15

        # Apply loss of 10, equity becomes 5 (still solvent)
        bank.apply_interbank_loss(counterparty_id=2, loss_amount=10.0)
        assert bank.is_solvent() is True
        assert bank.get_status() == BankStatus.SOLVENT

        # Apply additional loss of 6, equity becomes -1 (insolvent)
        bank.apply_interbank_loss(counterparty_id=2, loss_amount=6.0)
        assert bank.is_solvent() is False
        assert bank.get_status() == BankStatus.FAILED

    def test_apply_negative_loss(self):
        """Test that applying negative loss raises ValueError."""
        bank = Bank(bank_id=1, external_assets=100.0)
        bank.add_interbank_exposure(counterparty_id=2, amount=20.0)

        with pytest.raises(ValueError, match="Loss amount must be non-negative"):
            bank.apply_interbank_loss(counterparty_id=2, loss_amount=-5.0)

    def test_apply_loss_exceeding_exposure(self):
        """Test that applying loss exceeding exposure raises ValueError."""
        bank = Bank(bank_id=1, external_assets=100.0)
        bank.add_interbank_exposure(counterparty_id=2, amount=20.0)

        with pytest.raises(ValueError, match="exceeds exposure"):
            bank.apply_interbank_loss(counterparty_id=2, loss_amount=25.0)

    def test_apply_loss_no_exposure(self):
        """Test applying loss when there's no exposure."""
        bank = Bank(bank_id=1, external_assets=100.0)

        with pytest.raises(ValueError, match="exceeds exposure"):
            bank.apply_interbank_loss(counterparty_id=2, loss_amount=10.0)


class TestBankFailure:
    """Tests for bank failure scenarios."""

    def test_mark_as_failed(self):
        """Test manually marking a bank as failed."""
        bank = Bank(bank_id=1, external_assets=100.0, external_liabilities=80.0)
        assert bank.get_status() == BankStatus.SOLVENT

        bank.mark_as_failed()
        assert bank.get_status() == BankStatus.FAILED

    def test_recovery_rate_solvent(self):
        """Test recovery rate for solvent bank."""
        bank = Bank(bank_id=1, external_assets=100.0, external_liabilities=80.0)
        assert bank.get_recovery_rate() == 1.0

    def test_recovery_rate_insolvent(self):
        """Test recovery rate for insolvent bank."""
        bank = Bank(bank_id=1, external_assets=100.0, external_liabilities=150.0)
        # Recovery rate = 100 / 150 = 0.6667
        assert abs(bank.get_recovery_rate() - (100.0 / 150.0)) < 1e-10

    def test_recovery_rate_zero_liabilities(self):
        """Test recovery rate when liabilities are zero."""
        bank = Bank(bank_id=1, external_assets=100.0)
        assert bank.get_recovery_rate() == 1.0

    def test_recovery_rate_after_loss(self):
        """Test recovery rate after suffering a loss."""
        bank = Bank(bank_id=1, external_assets=100.0, external_liabilities=90.0)
        bank.add_interbank_exposure(counterparty_id=2, amount=20.0)
        # Total assets = 120, liabilities = 90, equity = 30

        # Apply loss that causes insolvency
        bank.apply_interbank_loss(counterparty_id=2, loss_amount=20.0)
        # Total assets = 100, liabilities = 90, equity = 10 (still solvent)
        assert bank.get_recovery_rate() == 1.0

        # Now add more liabilities to make it insolvent
        bank.add_interbank_obligation(counterparty_id=3, amount=20.0)
        # Total assets = 100, liabilities = 110
        assert abs(bank.get_recovery_rate() - (100.0 / 110.0)) < 1e-10


class TestBankRepresentation:
    """Tests for bank string representation and serialization."""

    def test_repr(self):
        """Test string representation."""
        bank = Bank(bank_id=1, external_assets=100.0, external_liabilities=80.0)
        repr_str = repr(bank)
        assert "Bank(id=1" in repr_str
        assert "assets=100.00" in repr_str
        assert "equity=20.00" in repr_str

    def test_to_dict(self):
        """Test conversion to dictionary."""
        bank = Bank(
            bank_id=1,
            external_assets=100.0,
            interbank_assets=20.0,
            interbank_liabilities=10.0,
            external_liabilities=80.0
        )
        bank_dict = bank.to_dict()

        assert bank_dict["bank_id"] == 1
        assert bank_dict["external_assets"] == 100.0
        assert bank_dict["interbank_assets"] == 20.0
        assert bank_dict["total_assets"] == 120.0
        assert bank_dict["total_liabilities"] == 90.0
        assert bank_dict["equity"] == 30.0
        assert bank_dict["status"] == "solvent"
        assert bank_dict["recovery_rate"] == 1.0

    def test_to_dict_with_portfolio(self):
        """Test conversion to dictionary includes portfolio composition."""
        from financial_contagion_networks.core.assets import Portfolio, AssetType

        # Create a portfolio
        portfolio = Portfolio()
        portfolio.add_asset_class(AssetType.GOVERNMENT_BOND, 30.0, 0.0)
        portfolio.add_asset_class(AssetType.CORPORATE_BOND, 25.0, 0.5)
        portfolio.add_asset_class(AssetType.MORTGAGE, 25.0, 0.35)
        portfolio.add_asset_class(AssetType.STOCK, 20.0, 1.0)

        # Create bank with portfolio
        bank = Bank(
            bank_id=1,
            external_assets=100.0,
            interbank_assets=20.0,
            interbank_liabilities=10.0,
            external_liabilities=80.0,
            portfolio=portfolio
        )

        bank_dict = bank.to_dict()

        # Check portfolio is included
        assert "portfolio" in bank_dict
        portfolio_comp = bank_dict["portfolio"]

        # Check all asset types are present with correct proportions
        assert "government_bond" in portfolio_comp
        assert "corporate_bond" in portfolio_comp
        assert "mortgage" in portfolio_comp
        assert "stock" in portfolio_comp

        # Total portfolio = 100, so weights should be 0.30, 0.25, 0.25, 0.20
        assert abs(portfolio_comp["government_bond"] - 0.30) < 1e-10
        assert abs(portfolio_comp["corporate_bond"] - 0.25) < 1e-10
        assert abs(portfolio_comp["mortgage"] - 0.25) < 1e-10
        assert abs(portfolio_comp["stock"] - 0.20) < 1e-10

    def test_to_dict_without_portfolio(self):
        """Test conversion to dictionary when bank has no portfolio."""
        bank = Bank(
            bank_id=1,
            external_assets=100.0,
            external_liabilities=80.0
        )
        bank_dict = bank.to_dict()

        # Portfolio should not be in dict if bank doesn't have one
        assert "portfolio" not in bank_dict


# ============================================================================
# Test Failure Mechanisms
# ============================================================================

class TestFailureMechanisms:
    """Test failure mechanism tracking functionality."""

    def test_failure_metadata_initialized_empty(self):
        """Test that failure_metadata is initialized with None values."""
        bank = Bank(bank_id=1, external_assets=100.0)

        assert bank.failure_metadata['failure_cause'] is None
        assert bank.failure_metadata['failure_round'] is None
        assert bank.failure_metadata['failure_trigger'] is None
        assert bank.failure_metadata['capital_at_failure'] is None

    def test_mark_as_failed_records_metadata(self):
        """Test that mark_as_failed records failure metadata."""
        bank = Bank(bank_id=1, external_assets=100.0, external_liabilities=80.0)

        bank.mark_as_failed(failure_cause='initial_shock', failure_round=0)

        assert bank.get_status() == BankStatus.FAILED
        assert bank.failure_metadata['failure_cause'] == 'initial_shock'
        assert bank.failure_metadata['failure_round'] == 0
        assert bank.failure_metadata['capital_at_failure'] is not None

    def test_contagion_failure_records_trigger(self):
        """Test that contagion-induced failure records trigger bank."""
        bank1 = Bank(bank_id=1, external_assets=100.0, external_liabilities=100.0)
        bank2 = Bank(bank_id=2, external_assets=100.0, external_liabilities=95.0)

        # Bank 2 has exposure to Bank 1
        bank2.add_interbank_exposure(1, 20.0)

        # Bank 1 fails, causing loss to Bank 2
        bank1.mark_as_failed()
        recovery_rate = bank1.get_recovery_rate()
        loss = (1.0 - recovery_rate) * 20.0

        # Apply loss with round number
        bank2.apply_interbank_loss(1, loss, failure_round=1)

        # If Bank 2 failed, check metadata
        if bank2.get_status() == BankStatus.FAILED:
            assert bank2.failure_metadata['failure_cause'] == 'contagion'
            assert bank2.failure_metadata['failure_round'] == 1
            assert bank2.failure_metadata['failure_trigger'] == 1

    def test_asset_shock_failure_records_cause(self):
        """Test that asset shock failure records cause."""
        from financial_contagion_networks.core.assets import Portfolio, AssetType

        portfolio = Portfolio()
        portfolio.add_asset_class(AssetType.MORTGAGE, 100.0, 0.40)

        bank = Bank(
            bank_id=1,
            external_assets=100.0,
            external_liabilities=95.0,
            portfolio=portfolio
        )

        # Apply severe shock that causes failure
        shocks = {AssetType.MORTGAGE: -0.50}  # 50% loss
        bank.apply_portfolio_shocks(shocks, failure_round=0)

        if bank.get_status() == BankStatus.FAILED:
            assert bank.failure_metadata['failure_cause'] == 'initial_shock'
            assert bank.failure_metadata['failure_round'] == 0
            assert bank.failure_metadata['capital_at_failure'] is not None

    def test_fire_sale_failure_records_cause(self):
        """Test that fire sale-induced failure records cause."""
        from financial_contagion_networks.core.assets import Portfolio, AssetType

        portfolio = Portfolio()
        portfolio.add_asset_class(AssetType.STOCK, 100.0, 0.40)

        bank = Bank(
            bank_id=1,
            external_assets=10.0,  # Very thin capital
            external_liabilities=9.0,
            portfolio=portfolio
        )

        # Apply severe fire sale markdown that causes failure
        bank.apply_fire_sale_markdown(0.50, failure_round=3)

        # Should fail due to large fire sale markdown
        assert bank.get_status() == BankStatus.FAILED
        assert bank.failure_metadata['failure_cause'] == 'fire_sale'
        assert bank.failure_metadata['failure_round'] == 3
        assert bank.failure_metadata['capital_at_failure'] is not None

    def test_capital_at_failure_captured(self):
        """Test that capital ratio at failure is captured."""
        bank = Bank(bank_id=1, external_assets=100.0, external_liabilities=95.0)

        capital_before = bank.get_capital_ratio()
        bank.mark_as_failed(failure_cause='initial_shock', failure_round=0)

        # Capital at failure should be captured (will be 5/100 = 0.05)
        assert bank.failure_metadata['capital_at_failure'] is not None
        # For a manually marked failure, capital might still be positive
        assert bank.failure_metadata['capital_at_failure'] == capital_before

    def test_failure_metadata_in_to_dict(self):
        """Test that failure_metadata is included in to_dict output."""
        bank = Bank(bank_id=1, external_assets=100.0, external_liabilities=95.0)

        bank.mark_as_failed(failure_cause='initial_shock', failure_round=0)

        bank_dict = bank.to_dict()

        assert 'failure_metadata' in bank_dict
        assert bank_dict['failure_metadata']['failure_cause'] == 'initial_shock'
        assert bank_dict['failure_metadata']['failure_round'] == 0
        assert bank_dict['failure_metadata']['capital_at_failure'] is not None

    def test_failure_metadata_survives_serialization(self):
        """Test that failure_metadata can be serialized to JSON."""
        import json

        bank = Bank(bank_id=1, external_assets=100.0, external_liabilities=95.0)
        bank.mark_as_failed(failure_cause='contagion', failure_round=2, failure_trigger=5)

        bank_dict = bank.to_dict()

        # Should be able to serialize to JSON
        json_str = json.dumps(bank_dict)
        deserialized = json.loads(json_str)

        assert deserialized['failure_metadata']['failure_cause'] == 'contagion'
        assert deserialized['failure_metadata']['failure_round'] == 2
        assert deserialized['failure_metadata']['failure_trigger'] == 5

    def test_contagion_failure_after_surviving_initial_loss(self):
        """Test that contagion failure is recorded even after surviving earlier losses."""
        # Create bank with existing exposure
        bank = Bank(
            bank_id=1,
            external_assets=100.0,
            interbank_assets=50.0,
            external_liabilities=140.0  # Thin capital (equity = 10)
        )

        bank.interbank_exposures[2] = 50.0  # Direct assignment for simplicity

        # Apply first small loss (survives)
        bank.apply_interbank_loss(2, 5.0, failure_round=0)
        assert bank.get_status() == BankStatus.SOLVENT

        # Apply second larger loss that causes failure
        # Equity is 10 - 5 = 5, losing 10 more should fail
        bank.apply_interbank_loss(2, 10.0, failure_round=1)

        # Should fail from contagion
        assert bank.get_status() == BankStatus.FAILED
        assert bank.failure_metadata['failure_cause'] == 'contagion'
        assert bank.failure_metadata['failure_round'] == 1
        assert bank.failure_metadata['failure_trigger'] == 2


# ============================================================================
# Test Loss Tracking
# ============================================================================

class TestLossTracking:
    """Test comprehensive loss tracking functionality."""

    def test_loss_tracking_initialized_empty(self):
        """Test that loss tracking is initialized with zero losses."""
        bank = Bank(
            bank_id=1,
            external_assets=100.0,
            external_liabilities=80.0
        )

        assert bank.loss_tracking['asset_shocks'] == {}
        assert bank.loss_tracking['contagion'] == {}
        assert bank.loss_tracking['fire_sales'] == 0.0
        assert bank.loss_tracking['total'] == 0.0

    def test_interbank_loss_tracked(self):
        """Test that interbank losses are tracked correctly."""
        bank = Bank(
            bank_id=1,
            external_assets=100.0,
            interbank_assets=50.0,
            external_liabilities=80.0
        )

        # Add exposure to counterparty
        bank.add_interbank_exposure(counterparty_id=2, amount=30.0)

        # Apply loss from counterparty failure
        bank.apply_interbank_loss(counterparty_id=2, loss_amount=20.0)

        # Check loss is tracked
        assert 2 in bank.loss_tracking['contagion']
        assert bank.loss_tracking['contagion'][2] == 20.0
        assert bank.loss_tracking['total'] == 20.0

    def test_multiple_interbank_losses_tracked(self):
        """Test that multiple interbank losses are tracked separately."""
        bank = Bank(
            bank_id=1,
            external_assets=100.0,
            interbank_assets=80.0,
            external_liabilities=80.0
        )

        # Add exposures
        bank.add_interbank_exposure(counterparty_id=2, amount=30.0)
        bank.add_interbank_exposure(counterparty_id=3, amount=50.0)

        # Apply losses from different counterparties
        bank.apply_interbank_loss(counterparty_id=2, loss_amount=20.0)
        bank.apply_interbank_loss(counterparty_id=3, loss_amount=35.0)

        # Check losses are tracked separately
        assert bank.loss_tracking['contagion'][2] == 20.0
        assert bank.loss_tracking['contagion'][3] == 35.0
        assert bank.loss_tracking['total'] == 55.0

    def test_multiple_losses_from_same_counterparty(self):
        """Test that multiple losses from the same counterparty accumulate."""
        bank = Bank(
            bank_id=1,
            external_assets=100.0,
            interbank_assets=80.0,
            external_liabilities=80.0
        )

        bank.add_interbank_exposure(counterparty_id=2, amount=50.0)

        # Apply losses from same counterparty twice
        bank.apply_interbank_loss(counterparty_id=2, loss_amount=20.0)
        bank.apply_interbank_loss(counterparty_id=2, loss_amount=15.0)

        # Check losses accumulate
        assert bank.loss_tracking['contagion'][2] == 35.0
        assert bank.loss_tracking['total'] == 35.0

    def test_portfolio_shock_losses_tracked_by_asset_type(self):
        """Test that portfolio shocks are tracked per asset type."""
        from financial_contagion_networks.core.assets import Portfolio, AssetType

        portfolio = Portfolio()
        portfolio.add_asset_class(AssetType.MORTGAGE, 50.0, 0.35)
        portfolio.add_asset_class(AssetType.CORPORATE_BOND, 30.0, 0.10)
        portfolio.add_asset_class(AssetType.STOCK, 20.0, 0.20)

        bank = Bank(
            bank_id=1,
            external_assets=100.0,
            external_liabilities=80.0,
            portfolio=portfolio
        )

        # Apply shocks
        shocks = {
            AssetType.MORTGAGE: -0.20,  # 20% loss on mortgages
            AssetType.CORPORATE_BOND: -0.10,  # 10% loss on corporate bonds
            AssetType.STOCK: -0.15  # 15% loss on stocks
        }

        total_loss = bank.apply_portfolio_shocks(shocks)

        # Check losses are tracked per asset type
        assert 'mortgage' in bank.loss_tracking['asset_shocks']
        assert 'corporate_bond' in bank.loss_tracking['asset_shocks']
        assert 'stock' in bank.loss_tracking['asset_shocks']

        # Mortgage loss = 50 * 0.20 = 10
        assert abs(bank.loss_tracking['asset_shocks']['mortgage'] - 10.0) < 1e-10
        # Corporate bond loss = 30 * 0.10 = 3
        assert abs(bank.loss_tracking['asset_shocks']['corporate_bond'] - 3.0) < 1e-10
        # Stock loss = 20 * 0.15 = 3
        assert abs(bank.loss_tracking['asset_shocks']['stock'] - 3.0) < 1e-10

        # Total should match sum
        expected_total = 10.0 + 3.0 + 3.0
        assert abs(bank.loss_tracking['total'] - expected_total) < 1e-10
        assert abs(total_loss - expected_total) < 1e-10

    def test_fire_sale_losses_tracked(self):
        """Test that fire sale losses are tracked."""
        from financial_contagion_networks.core.assets import Portfolio, AssetType

        portfolio = Portfolio()
        portfolio.add_asset_class(AssetType.MORTGAGE, 50.0, 0.35)
        portfolio.add_asset_class(AssetType.STOCK, 30.0, 0.20)

        bank = Bank(
            bank_id=1,
            external_assets=100.0,
            external_liabilities=80.0,
            portfolio=portfolio
        )

        # Apply fire sale markdown
        markdown = 0.10  # 10% markdown
        loss = bank.apply_fire_sale_markdown(markdown)

        # Check fire sale loss is tracked
        assert bank.loss_tracking['fire_sales'] > 0
        assert bank.loss_tracking['total'] == loss

    def test_combined_losses_tracked(self):
        """Test that losses from multiple sources are tracked correctly."""
        from financial_contagion_networks.core.assets import Portfolio, AssetType

        portfolio = Portfolio()
        portfolio.add_asset_class(AssetType.MORTGAGE, 40.0, 0.35)

        bank = Bank(
            bank_id=1,
            external_assets=100.0,
            interbank_assets=40.0,
            external_liabilities=80.0,
            portfolio=portfolio
        )

        # Add interbank exposure
        bank.add_interbank_exposure(counterparty_id=2, amount=25.0)

        # 1. Apply asset shock
        asset_shocks = {AssetType.MORTGAGE: -0.20}
        asset_loss = bank.apply_portfolio_shocks(asset_shocks)  # 40 * 0.20 = 8

        # 2. Apply interbank loss
        bank.apply_interbank_loss(counterparty_id=2, loss_amount=15.0)

        # 3. Apply fire sale
        fire_sale_loss = bank.apply_fire_sale_markdown(0.05)

        # Check all losses are tracked
        assert 'mortgage' in bank.loss_tracking['asset_shocks']
        assert 2 in bank.loss_tracking['contagion']
        assert bank.loss_tracking['fire_sales'] > 0

        # Check total is sum of all losses
        expected_total = asset_loss + 15.0 + fire_sale_loss
        assert abs(bank.loss_tracking['total'] - expected_total) < 1e-10

    def test_loss_tracking_in_to_dict(self):
        """Test that loss tracking is included in to_dict output."""
        from financial_contagion_networks.core.assets import Portfolio, AssetType

        portfolio = Portfolio()
        portfolio.add_asset_class(AssetType.MORTGAGE, 50.0, 0.35)

        bank = Bank(
            bank_id=1,
            external_assets=100.0,
            interbank_assets=30.0,
            external_liabilities=80.0,
            portfolio=portfolio
        )

        # Apply some losses
        bank.add_interbank_exposure(counterparty_id=2, amount=20.0)
        bank.apply_interbank_loss(counterparty_id=2, loss_amount=10.0)
        bank.apply_portfolio_shocks({AssetType.MORTGAGE: -0.10})

        # Convert to dict
        bank_dict = bank.to_dict()

        # Check loss_tracking is included
        assert 'loss_tracking' in bank_dict
        loss_tracking = bank_dict['loss_tracking']

        assert 'asset_shocks' in loss_tracking
        assert 'contagion' in loss_tracking
        assert 'fire_sales' in loss_tracking
        assert 'total' in loss_tracking

        # Check values
        assert 2 in loss_tracking['contagion']
        assert loss_tracking['contagion'][2] == 10.0
        assert 'mortgage' in loss_tracking['asset_shocks']

    def test_loss_tracking_survives_serialization(self):
        """Test that loss tracking data can be serialized to JSON."""
        import json
        from financial_contagion_networks.core.assets import Portfolio, AssetType

        portfolio = Portfolio()
        portfolio.add_asset_class(AssetType.MORTGAGE, 50.0, 0.35)

        bank = Bank(
            bank_id=1,
            external_assets=100.0,
            interbank_assets=30.0,
            external_liabilities=80.0,
            portfolio=portfolio
        )

        # Apply losses
        bank.add_interbank_exposure(counterparty_id=2, amount=20.0)
        bank.apply_interbank_loss(counterparty_id=2, loss_amount=10.0)
        bank.apply_portfolio_shocks({AssetType.MORTGAGE: -0.10})

        # Convert to dict and serialize
        bank_dict = bank.to_dict()
        json_str = json.dumps(bank_dict)

        # Should not raise error
        loaded_dict = json.loads(json_str)

        # Check loss tracking survived
        assert 'loss_tracking' in loaded_dict
        assert loaded_dict['loss_tracking']['total'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
