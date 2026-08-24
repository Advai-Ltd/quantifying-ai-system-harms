"""
Bank module for financial contagion modeling.

This module implements the Bank class representing a financial institution
with a balance sheet and methods for tracking solvency and contagion effects.
"""

from typing import Dict, Optional
from enum import Enum
try:
    from financial_contagion_networks.core.assets import Portfolio, AssetType
    PORTFOLIO_AVAILABLE = True
except ImportError:
    PORTFOLIO_AVAILABLE = False


class BankStatus(Enum):
    """Enumeration of possible bank states."""
    SOLVENT = "solvent"
    FAILED = "failed"


class Bank:
    """
    Represents a financial institution with a balance sheet.

    The balance sheet consists of:
    - Assets:
        - external_assets: Loans to non-banks, securities, etc.
        - interbank_assets: Claims on other banks (loans to other banks)
    - Liabilities:
        - interbank_liabilities: Obligations to other banks (borrowing from other banks)
        - external_liabilities: Deposits and other external obligations
    - Equity: Capital buffer (assets - liabilities)

    A bank fails when its total assets fall below its total liabilities (equity <= 0).
    """

    def __init__(
        self,
        bank_id: int,
        external_assets: float,
        interbank_assets: float = 0.0,
        interbank_liabilities: float = 0.0,
        external_liabilities: float = 0.0,
        portfolio: Optional['Portfolio'] = None
    ):
        """
        Initialize a Bank.

        Args:
            bank_id: Unique identifier for the bank
            external_assets: Non-interbank assets (must be positive)
            interbank_assets: Total claims on other banks (must be non-negative)
            interbank_liabilities: Total obligations to other banks (must be non-negative)
            external_liabilities: External obligations like deposits (must be non-negative)
            portfolio: Optional portfolio of diversified assets (for advanced simulations)

        Raises:
            ValueError: If any parameter has an invalid value
        """
        if bank_id < 0:
            raise ValueError("bank_id must be non-negative")
        if external_assets <= 0:
            raise ValueError("external_assets must be positive")
        if interbank_assets < 0:
            raise ValueError("interbank_assets must be non-negative")
        if interbank_liabilities < 0:
            raise ValueError("interbank_liabilities must be non-negative")
        if external_liabilities < 0:
            raise ValueError("external_liabilities must be non-negative")

        self.bank_id = bank_id
        self.external_assets = external_assets
        self.interbank_assets = interbank_assets
        self.interbank_liabilities = interbank_liabilities
        self.external_liabilities = external_liabilities

        # Track exposures to specific banks
        self.interbank_exposures: Dict[int, float] = {}  # bank_id -> exposure amount
        self.interbank_obligations: Dict[int, float] = {}  # bank_id -> obligation amount

        # Track if interbank assets/liabilities were pre-allocated (to avoid double-counting)
        self._interbank_preallocated = (interbank_assets > 0 or interbank_liabilities > 0)

        # Portfolio (optional, for advanced simulations)
        self.portfolio = portfolio

        # Status tracking
        self._status = BankStatus.SOLVENT
        self._initial_equity = self.get_equity()

        # Loss tracking for comprehensive data capture
        # Tracks losses from each source: asset shocks, interbank contagion, fire sales
        self.loss_tracking = {
            'asset_shocks': {},  # {asset_type: loss_amount}
            'contagion': {},     # {counterparty_id: loss_amount}
            'fire_sales': 0.0,
            'total': 0.0
        }

        # Failure mechanism tracking
        # Records exactly how and when this bank failed
        self.failure_metadata = {
            'failure_cause': None,      # "initial_shock" | "contagion" | "fire_sale"
            'failure_round': None,      # Which round did it fail?
            'failure_trigger': None,    # Bank ID that triggered failure (for contagion)
            'capital_at_failure': None  # Capital ratio at moment of failure
        }

    def get_total_assets(self) -> float:
        """Calculate total assets."""
        return self.external_assets + self.interbank_assets

    def get_total_liabilities(self) -> float:
        """Calculate total liabilities."""
        return self.interbank_liabilities + self.external_liabilities

    def get_equity(self) -> float:
        """Calculate equity (capital buffer)."""
        return self.get_total_assets() - self.get_total_liabilities()

    def get_leverage_ratio(self) -> float:
        """Calculate leverage ratio (total assets / equity)."""
        equity = self.get_equity()
        if equity <= 0:
            return float('inf')
        return self.get_total_assets() / equity

    def get_capital_ratio(self) -> float:
        """Calculate capital ratio (equity / total assets)."""
        total_assets = self.get_total_assets()
        if total_assets == 0:
            return 0.0
        return self.get_equity() / total_assets

    def is_solvent(self) -> bool:
        """Check if the bank is solvent (equity > 0)."""
        return self.get_equity() > 0

    def get_status(self) -> BankStatus:
        """Get the current status of the bank."""
        return self._status

    def add_interbank_exposure(self, counterparty_id: int, amount: float) -> None:
        """
        Add an exposure to another bank.

        Args:
            counterparty_id: ID of the counterparty bank
            amount: Amount of the exposure (must be positive)

        Raises:
            ValueError: If amount is not positive or bank lends to itself
        """
        if counterparty_id == self.bank_id:
            raise ValueError("Bank cannot have exposure to itself")
        if amount <= 0:
            raise ValueError("Exposure amount must be positive")

        self.interbank_exposures[counterparty_id] = \
            self.interbank_exposures.get(counterparty_id, 0.0) + amount
        # Only increment total if not pre-allocated (to avoid double-counting)
        if not self._interbank_preallocated:
            # Building from scratch: increment for each exposure
            self.interbank_assets += amount
        # else: Pre-allocated budget, exposures redistribute it (don't increment)

    def add_interbank_obligation(self, counterparty_id: int, amount: float) -> None:
        """
        Add an obligation to another bank.

        Args:
            counterparty_id: ID of the counterparty bank
            amount: Amount of the obligation (must be positive)

        Raises:
            ValueError: If amount is not positive or bank borrows from itself
        """
        if counterparty_id == self.bank_id:
            raise ValueError("Bank cannot have obligation to itself")
        if amount <= 0:
            raise ValueError("Obligation amount must be positive")

        self.interbank_obligations[counterparty_id] = \
            self.interbank_obligations.get(counterparty_id, 0.0) + amount
        # Only increment total if not pre-allocated (to avoid double-counting)
        if not self._interbank_preallocated:
            # Building from scratch: increment for each obligation
            self.interbank_liabilities += amount
        # else: Pre-allocated budget, obligations redistribute it (don't increment)

    def apply_interbank_loss(self, counterparty_id: int, loss_amount: float,
                            failure_round: int = None) -> None:
        """
        Apply a loss from a failed counterparty.

        When a counterparty bank fails, this bank loses part or all of its exposure
        to that counterparty, reducing its interbank assets.

        Args:
            counterparty_id: ID of the failed counterparty
            loss_amount: Amount of the loss (must be non-negative)
            failure_round: Round number when this loss is applied (for failure tracking)

        Raises:
            ValueError: If loss amount is negative or exceeds exposure
        """
        if loss_amount < 0:
            raise ValueError("Loss amount must be non-negative")

        exposure = self.interbank_exposures.get(counterparty_id, 0.0)
        if loss_amount > exposure:
            raise ValueError(f"Loss amount {loss_amount} exceeds exposure {exposure}")

        # Reduce interbank assets
        self.interbank_assets -= loss_amount
        self.interbank_exposures[counterparty_id] -= loss_amount

        # Track loss for comprehensive data capture
        if counterparty_id not in self.loss_tracking['contagion']:
            self.loss_tracking['contagion'][counterparty_id] = 0.0
        self.loss_tracking['contagion'][counterparty_id] += loss_amount
        self.loss_tracking['total'] += loss_amount

        # Update status if bank becomes insolvent
        if not self.is_solvent() and self._status == BankStatus.SOLVENT:
            self._status = BankStatus.FAILED
            # Record failure metadata for contagion-induced failure
            self.failure_metadata['failure_cause'] = 'contagion'
            if failure_round is not None:
                self.failure_metadata['failure_round'] = failure_round
            self.failure_metadata['failure_trigger'] = counterparty_id
            self.failure_metadata['capital_at_failure'] = self.get_capital_ratio()

    def mark_as_failed(self, failure_cause: str = None, failure_round: int = None,
                      failure_trigger: int = None) -> None:
        """
        Mark the bank as failed and record failure metadata.

        Args:
            failure_cause: Cause of failure ("initial_shock", "contagion", "fire_sale")
            failure_round: Round number when failure occurred
            failure_trigger: Bank ID that triggered failure (for contagion)
        """
        self._status = BankStatus.FAILED

        # Record failure metadata
        if failure_cause is not None:
            self.failure_metadata['failure_cause'] = failure_cause
        if failure_round is not None:
            self.failure_metadata['failure_round'] = failure_round
        if failure_trigger is not None:
            self.failure_metadata['failure_trigger'] = failure_trigger

        # Capture capital ratio at failure
        self.failure_metadata['capital_at_failure'] = self.get_capital_ratio()

    def get_recovery_rate(self) -> float:
        """
        Calculate the recovery rate for creditors if the bank fails.

        If the bank is solvent, recovery rate is 100%.
        If insolvent, recovery rate is total_assets / total_liabilities.

        Returns:
            Recovery rate between 0 and 1
        """
        if self.is_solvent():
            return 1.0

        total_liabilities = self.get_total_liabilities()
        if total_liabilities == 0:
            return 1.0

        return min(1.0, self.get_total_assets() / total_liabilities)

    def get_interbank_recovery_rate(self, priority: bool = False) -> float:
        """
        Calculate the recovery rate specifically for interbank creditors.

        If priority=False: Returns same as get_recovery_rate() (all creditors equal)
        If priority=True: Interbank creditors are junior to external creditors

        Priority structure (when enabled):
        1. External liabilities (deposits, senior debt) get paid first
        2. Interbank liabilities get residual (if any)

        Example with priority=True:
        - Assets: $95, External liabilities: $90, Interbank liabilities: $10
        - External creditors get: $90 (100% recovery)
        - Interbank creditors get: $5 out of $10 (50% recovery)

        Args:
            priority: If True, apply priority of claims (interbank is junior)

        Returns:
            Recovery rate for interbank creditors (0 to 1)
        """
        # Solvent banks always have 100% recovery
        if self.is_solvent():
            return 1.0

        # No interbank liabilities = 100% recovery (nothing to recover)
        if self.interbank_liabilities == 0:
            return 1.0

        # Without priority: all creditors get same rate
        if not priority:
            return self.get_recovery_rate()

        # WITH PRIORITY: Interbank creditors are junior
        # Step 1: External creditors get paid first (up to their full claim)
        total_assets = self.get_total_assets()

        # Step 2: Residual after external creditors paid
        # External creditors can claim up to min(external_liabilities, total_assets)
        paid_to_external = min(self.external_liabilities, total_assets)
        residual = max(0, total_assets - paid_to_external)

        # Step 3: Interbank creditors split the residual
        if self.interbank_liabilities == 0:
            return 1.0

        interbank_recovery = residual / self.interbank_liabilities

        return min(1.0, interbank_recovery)

    # Portfolio-based methods (for advanced simulations)

    def has_portfolio(self) -> bool:
        """Check if bank has a portfolio (for advanced simulations)."""
        return self.portfolio is not None

    def apply_portfolio_shock(self, asset_type: 'AssetType', shock: float) -> float:
        """
        Apply a shock to specific asset class in portfolio.

        Args:
            asset_type: Type of asset to shock
            shock: Percentage loss (e.g., -0.20 = 20% loss)

        Returns:
            Loss amount

        Raises:
            ValueError: If bank doesn't have a portfolio
        """
        if not self.has_portfolio():
            raise ValueError("Bank does not have a portfolio")

        loss = self.portfolio.apply_shock(asset_type, shock)

        # Reduce external assets by loss amount
        self.external_assets = max(0.0, self.external_assets - loss)

        # Update status if bank becomes insolvent
        if not self.is_solvent() and self._status == BankStatus.SOLVENT:
            self._status = BankStatus.FAILED

        return loss

    def apply_portfolio_shocks(self, shocks: Dict['AssetType', float],
                              failure_round: int = 0) -> float:
        """
        Apply multiple correlated shocks to portfolio.

        Args:
            shocks: Dict mapping asset types to shock percentages
            failure_round: Round number when shocks are applied (for failure tracking)

        Returns:
            Total loss amount
        """
        if not self.has_portfolio():
            raise ValueError("Bank does not have a portfolio")

        # Apply shocks individually to track per-asset-type losses
        total_loss = 0.0
        for asset_type, shock in shocks.items():
            loss = self.portfolio.apply_shock(asset_type, shock)
            total_loss += loss

            # Track loss per asset type for comprehensive data capture
            asset_type_str = asset_type.value if hasattr(asset_type, 'value') else str(asset_type)
            self.loss_tracking['asset_shocks'][asset_type_str] = loss
            self.loss_tracking['total'] += loss

        # Reduce external assets by total loss
        self.external_assets = max(0.0, self.external_assets - total_loss)

        # Update status if bank becomes insolvent
        if not self.is_solvent() and self._status == BankStatus.SOLVENT:
            self._status = BankStatus.FAILED
            # Record failure metadata for asset shock-induced failure
            self.failure_metadata['failure_cause'] = 'initial_shock'
            self.failure_metadata['failure_round'] = failure_round
            self.failure_metadata['capital_at_failure'] = self.get_capital_ratio()

        return total_loss

    def apply_fire_sale_markdown(self, markdowns: Dict['AssetType', float], failure_round: int = None) -> float:
        """
        Apply asset-specific fire sale markdowns to portfolio.

        When banks fail and liquidate assets, they depress prices for those specific assets.

        Args:
            markdowns: Dict mapping asset types to markdown percentages
                      (e.g., {AssetType.MORTGAGE: 0.10, AssetType.STOCK: 0.05})
            failure_round: Round number when markdown is applied (for failure tracking)

        Returns:
            Total loss amount from all markdowns
        """
        if not self.has_portfolio():
            return 0.0

        loss = self.portfolio.apply_fire_sale_markdown(markdowns)

        # Track fire sale loss for comprehensive data capture
        self.loss_tracking['fire_sales'] += loss
        self.loss_tracking['total'] += loss

        # Reduce external assets by loss amount
        self.external_assets = max(0.0, self.external_assets - loss)

        # Update status if bank becomes insolvent
        if not self.is_solvent() and self._status == BankStatus.SOLVENT:
            self._status = BankStatus.FAILED
            # Record failure metadata for fire sale-induced failure
            self.failure_metadata['failure_cause'] = 'fire_sale'
            if failure_round is not None:
                self.failure_metadata['failure_round'] = failure_round
            self.failure_metadata['capital_at_failure'] = self.get_capital_ratio()

        return loss

    def get_portfolio_composition(self) -> Optional[Dict['AssetType', float]]:
        """Get portfolio composition as percentages."""
        if not self.has_portfolio():
            return None
        return self.portfolio.get_composition()

    def __repr__(self) -> str:
        """String representation of the bank."""
        return (f"Bank(id={self.bank_id}, "
                f"assets={self.get_total_assets():.2f}, "
                f"liabilities={self.get_total_liabilities():.2f}, "
                f"equity={self.get_equity():.2f}, "
                f"status={self._status.value})")

    def to_dict(self) -> Dict:
        """Convert bank state to dictionary."""
        total_assets = self.get_total_assets()

        result = {
            "bank_id": self.bank_id,
            "external_assets": self.external_assets,
            "interbank_assets": self.interbank_assets,
            "interbank_liabilities": self.interbank_liabilities,
            "external_liabilities": self.external_liabilities,
            "total_assets": total_assets,
            "total_liabilities": self.get_total_liabilities(),
            "equity": self.get_equity(),
            "capital_ratio": self.get_capital_ratio(),
            "status": self._status.value,
            "recovery_rate": self.get_recovery_rate(),
            "loss_tracking": self.loss_tracking,  # Comprehensive loss attribution
            "failure_metadata": self.failure_metadata,  # How and when the bank failed
            # CRITICAL: Interbank assets fraction (Option A validation)
            "interbank_assets_fraction": self.interbank_assets / total_assets if total_assets > 0 else 0.0
        }

        # Add portfolio composition if available
        portfolio_comp = self.get_portfolio_composition()
        if portfolio_comp is not None:
            # Convert AssetType enum keys to strings for JSON serialization
            result["portfolio"] = {
                asset_type.value: weight
                for asset_type, weight in portfolio_comp.items()
            }

        return result
