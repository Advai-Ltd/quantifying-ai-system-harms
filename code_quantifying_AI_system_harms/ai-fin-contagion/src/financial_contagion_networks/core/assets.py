"""
Asset classes for financial instruments in the contagion model.

This module provides a single-period model where banks hold portfolios
of different asset types that can lose value in correlated shocks.
"""

from typing import Dict, Optional, Union
from enum import Enum
from dataclasses import dataclass


class AssetType(Enum):
    """Types of financial assets."""
    GOVERNMENT_BOND = "government_bond"
    CORPORATE_BOND = "corporate_bond"
    MORTGAGE = "mortgage"
    STOCK = "stock"
    INTERBANK_LOAN = "interbank_loan"


@dataclass
class AssetClass:
    """
    Represents a class of assets (not individual securities).

    In this single-period model, banks hold amounts of different asset classes
    that can experience correlated shocks.
    """
    asset_type: AssetType
    amount: float  # Face value / book value
    risk_weight: float = 1.0  # Risk weight for capital requirements (0-1)

    def get_value(self) -> float:
        """Get current value of assets in this class."""
        return self.amount

    def apply_shock(self, shock: float) -> float:
        """
        Apply a value shock to this asset class.

        Args:
            shock: Percentage loss (e.g., -0.20 = 20% loss)

        Returns:
            Loss amount
        """
        loss = abs(shock) * self.amount
        self.amount = max(0.0, self.amount * (1.0 + shock))
        return loss


class Portfolio:
    """
    Portfolio of financial assets held by a bank.

    Manages different asset classes and provides valuation and shock methods.
    """

    def __init__(self):
        """Initialize empty portfolio."""
        self.asset_classes: Dict[AssetType, AssetClass] = {}

    def add_asset_class(self, asset_type: AssetType, amount: float, risk_weight: float = 1.0) -> None:
        """
        Add or increase holdings in an asset class.

        Args:
            asset_type: Type of asset
            amount: Amount to add
            risk_weight: Risk weight for capital calculations
        """
        if asset_type in self.asset_classes:
            self.asset_classes[asset_type].amount += amount
        else:
            self.asset_classes[asset_type] = AssetClass(asset_type, amount, risk_weight)

    def get_total_value(self) -> float:
        """Get total portfolio value."""
        return sum(ac.get_value() for ac in self.asset_classes.values())

    def get_value_by_type(self, asset_type: AssetType) -> float:
        """Get value of specific asset class."""
        if asset_type in self.asset_classes:
            return self.asset_classes[asset_type].get_value()
        return 0.0

    def get_composition(self) -> Dict[AssetType, float]:
        """
        Get portfolio composition as percentages.

        Returns:
            Dict mapping asset type to percentage of portfolio
        """
        total = self.get_total_value()
        if total == 0:
            return {}

        return {
            asset_type: ac.get_value() / total
            for asset_type, ac in self.asset_classes.items()
        }

    def apply_shock(self, asset_type: AssetType, shock: float) -> float:
        """
        Apply shock to specific asset class.

        Args:
            asset_type: Asset class to shock
            shock: Percentage loss (negative)

        Returns:
            Total loss amount
        """
        if asset_type in self.asset_classes:
            return self.asset_classes[asset_type].apply_shock(shock)
        return 0.0

    def apply_shocks(self, shocks: Dict[AssetType, float]) -> float:
        """
        Apply multiple correlated shocks to different asset classes.

        Args:
            shocks: Dict mapping asset types to shock percentages

        Returns:
            Total loss amount across all asset classes
        """
        total_loss = 0.0
        for asset_type, shock in shocks.items():
            total_loss += self.apply_shock(asset_type, shock)
        return total_loss

    def apply_fire_sale_markdown(self, markdowns: Union[float, Dict[AssetType, float]]) -> float:
        """
        Apply asset-specific fire sale markdowns.

        When banks fail and liquidate assets, they depress prices for those specific assets.
        This method applies different markdowns to different asset types based on actual
        liquidation volumes.

        Args:
            markdowns: Either:
                      - Dict mapping asset types to specific markdown percentages
                        (e.g., {AssetType.MORTGAGE: 0.10, AssetType.STOCK: 0.05})
                      - Float for uniform markdown across all assets (backward compatibility)

        Returns:
            Total loss from all markdowns

        Example:
            If 5 banks with lots of mortgages fail, mortgages get marked down heavily,
            but government bonds (not sold) don't get marked down at all.
        """
        total_loss = 0.0

        # Backward compatibility: if markdowns is a float, apply to all tradable assets
        if isinstance(markdowns, (float, int)):
            uniform_markdown = float(markdowns)
            tradable_types = [
                AssetType.GOVERNMENT_BOND,
                AssetType.CORPORATE_BOND,
                AssetType.MORTGAGE,
                AssetType.STOCK
            ]
            for asset_type in tradable_types:
                if asset_type in self.asset_classes and uniform_markdown > 0:
                    loss = self.asset_classes[asset_type].apply_shock(-uniform_markdown)
                    total_loss += loss
        else:
            # New behavior: asset-specific markdowns
            for asset_type, markdown in markdowns.items():
                if asset_type in self.asset_classes and markdown > 0:
                    loss = self.asset_classes[asset_type].apply_shock(-markdown)
                    total_loss += loss

        return total_loss

    def get_risk_weighted_assets(self) -> float:
        """
        Calculate risk-weighted assets for capital requirements.

        Returns:
            Sum of assets weighted by risk weights
        """
        return sum(
            ac.amount * ac.risk_weight
            for ac in self.asset_classes.values()
        )

    def __repr__(self) -> str:
        return f"Portfolio(value={self.get_total_value():.2f}, classes={len(self.asset_classes)})"


# Risk weights for different asset classes (Basel-style)
STANDARD_RISK_WEIGHTS = {
    AssetType.GOVERNMENT_BOND: 0.0,  # Zero risk weight for sovereign bonds
    AssetType.CORPORATE_BOND: 0.5,   # 50% risk weight
    AssetType.MORTGAGE: 0.35,        # 35% risk weight (residential mortgages)
    AssetType.STOCK: 1.0,             # 100% risk weight
    AssetType.INTERBANK_LOAN: 0.2    # 20% risk weight
}
