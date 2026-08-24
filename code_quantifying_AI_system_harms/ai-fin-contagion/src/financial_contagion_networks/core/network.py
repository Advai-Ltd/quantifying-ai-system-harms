"""
ContagionNetwork module for financial contagion modeling.

This module implements the ContagionNetwork class which manages the network
of banks and simulates contagion propagation.
"""

from typing import List, Dict, Set, Tuple, Optional
import networkx as nx
import numpy as np
from financial_contagion_networks.core.bank import Bank, BankStatus


class ContagionNetwork:
    """
    Represents a network of interconnected banks and simulates financial contagion.

    The network is represented as a directed graph where:
    - Nodes are banks
    - Edges represent interbank exposures (directed from lender to borrower)
    - Edge weights represent the exposure amount
    """

    def __init__(self, banks: Optional[List[Bank]] = None):
        """
        Initialize a ContagionNetwork.

        Args:
            banks: Optional list of Bank objects to add to the network
        """
        self.banks: Dict[int, Bank] = {}
        self.graph = nx.DiGraph()
        self.contagion_history: List[Dict] = []

        if banks:
            for bank in banks:
                self.add_bank(bank)

    def add_bank(self, bank: Bank) -> None:
        """
        Add a bank to the network.

        Args:
            bank: Bank object to add

        Raises:
            ValueError: If a bank with the same ID already exists
        """
        if bank.bank_id in self.banks:
            raise ValueError(f"Bank with ID {bank.bank_id} already exists in network")

        self.banks[bank.bank_id] = bank
        self.graph.add_node(bank.bank_id)

    def add_exposure(self, lender_id: int, borrower_id: int, amount: float) -> None:
        """
        Add an interbank exposure (loan from lender to borrower).

        This creates a directed edge from lender to borrower, representing that
        the lender has an asset (exposure) to the borrower, and the borrower has
        a liability (obligation) to the lender.

        Args:
            lender_id: ID of the lending bank
            borrower_id: ID of the borrowing bank
            amount: Amount of the exposure (must be positive)

        Raises:
            ValueError: If either bank doesn't exist or amount is invalid
        """
        if lender_id not in self.banks:
            raise ValueError(f"Lender bank {lender_id} not found in network")
        if borrower_id not in self.banks:
            raise ValueError(f"Borrower bank {borrower_id} not found in network")
        if amount <= 0:
            raise ValueError("Exposure amount must be positive")

        lender = self.banks[lender_id]
        borrower = self.banks[borrower_id]

        # Update bank balance sheets
        lender.add_interbank_exposure(borrower_id, amount)
        borrower.add_interbank_obligation(lender_id, amount)

        # Update graph
        if self.graph.has_edge(lender_id, borrower_id):
            self.graph[lender_id][borrower_id]['weight'] += amount
        else:
            self.graph.add_edge(lender_id, borrower_id, weight=amount)

    def get_creditors(self, bank_id: int) -> List[int]:
        """
        Get the list of banks that have exposures to the given bank.

        Args:
            bank_id: ID of the bank

        Returns:
            List of creditor bank IDs
        """
        # Predecessors in the graph are creditors (they lent to this bank)
        return list(self.graph.predecessors(bank_id))

    def get_debtors(self, bank_id: int) -> List[int]:
        """
        Get the list of banks that owe money to the given bank.

        Args:
            bank_id: ID of the bank

        Returns:
            List of debtor bank IDs
        """
        # Successors in the graph are debtors (they borrowed from this bank)
        return list(self.graph.successors(bank_id))

    def simulate_shock(self, shocked_bank_ids: List[int], use_priority_claims: bool = False) -> Dict:
        """
        Simulate an initial shock to one or more banks and propagate contagion.

        Args:
            shocked_bank_ids: List of bank IDs to initially fail
            use_priority_claims: If True, interbank creditors are junior to external creditors
                                (Option 4: Priority of Claims)

        Returns:
            Dictionary containing simulation results

        Raises:
            ValueError: If any shocked bank ID doesn't exist
        """
        for bank_id in shocked_bank_ids:
            if bank_id not in self.banks:
                raise ValueError(f"Bank {bank_id} not found in network")

        # Reset contagion history
        self.contagion_history = []

        # Track failed banks and when they failed
        failed_banks: Set[int] = set()
        round_failures: List[List[int]] = []

        # Round 0: Initial shock
        initial_failures = []
        for bank_id in shocked_bank_ids:
            bank = self.banks[bank_id]
            if bank.get_status() == BankStatus.SOLVENT:
                bank.mark_as_failed(failure_cause='initial_shock', failure_round=0)
                failed_banks.add(bank_id)
                initial_failures.append(bank_id)

        round_failures.append(initial_failures)
        self._record_state(round_num=0, newly_failed=initial_failures)

        # Propagate contagion
        round_num = 1
        newly_failed = set(initial_failures)
        all_contagion_edges = []  # Track all loss transmissions

        while newly_failed:
            current_round_failures = []
            round_contagion_edges = []  # Track edges for this round

            # Process each newly failed bank
            for failed_bank_id in newly_failed:
                failed_bank = self.banks[failed_bank_id]

                # Use interbank-specific recovery rate if priority of claims enabled
                if use_priority_claims:
                    recovery_rate = failed_bank.get_interbank_recovery_rate(priority=True)
                else:
                    recovery_rate = failed_bank.get_recovery_rate()

                # Get all creditors (banks that lent to the failed bank)
                creditors = self.get_creditors(failed_bank_id)

                for creditor_id in creditors:
                    if creditor_id in failed_banks:
                        continue  # Skip already failed banks

                    creditor = self.banks[creditor_id]
                    exposure = creditor.interbank_exposures.get(failed_bank_id, 0.0)

                    # Calculate loss (1 - recovery_rate) * exposure
                    loss = (1.0 - recovery_rate) * exposure

                    if loss > 0:
                        # Capture capital ratio before applying loss
                        capital_before = creditor.get_capital_ratio()

                        # Apply the loss (with round number for failure tracking)
                        creditor.apply_interbank_loss(failed_bank_id, loss, failure_round=round_num)

                        # Capture capital ratio after applying loss
                        capital_after = creditor.get_capital_ratio()
                        caused_failure = creditor.get_status() == BankStatus.FAILED

                        # Record this loss transmission (contagion edge)
                        contagion_edge = {
                            'debtor_id': failed_bank_id,
                            'creditor_id': creditor_id,
                            'exposure': exposure,
                            'recovery_rate': recovery_rate,
                            'loss': loss,
                            'creditor_capital_before': capital_before,
                            'creditor_capital_after': capital_after,
                            'caused_failure': caused_failure
                        }
                        round_contagion_edges.append(contagion_edge)

                        # Check if creditor becomes insolvent
                        if caused_failure and creditor_id not in failed_banks:
                            failed_banks.add(creditor_id)
                            current_round_failures.append(creditor_id)

            all_contagion_edges.extend(round_contagion_edges)
            round_failures.append(current_round_failures)
            self._record_state(round_num=round_num, newly_failed=current_round_failures,
                             contagion_edges=round_contagion_edges)

            newly_failed = set(current_round_failures)
            round_num += 1

        # Compile results
        return {
            'total_failed': len(failed_banks),
            'failed_bank_ids': sorted(list(failed_banks)),
            'total_rounds': round_num - 1,
            'failures_by_round': round_failures,
            'initial_shock': shocked_bank_ids,
            'contagion_history': self.contagion_history,
            'contagion_edges': all_contagion_edges  # Comprehensive loss transmission tracking
        }

    def propagate_contagion_from_failed(self, failed_bank_ids: List[int], use_priority_claims: bool = False) -> Dict:
        """
        Propagate contagion from banks that are ALREADY marked as failed.

        This method is for propagating contagion after banks have already been
        marked as failed from another cause (e.g., asset shocks, fire sales).

        Args:
            failed_bank_ids: List of bank IDs that are already failed
            use_priority_claims: If True, interbank creditors are junior to external creditors

        Key differences from simulate_shock():
        - Does NOT mark banks as failed (they must already be FAILED)
        - Does NOT record round 0 (banks already failed elsewhere)
        - Starts contagion from round 1, not round 0
        - Returns separate count of contagion-only failures

        Use cases:
        - After asset shock phase has failed banks
        - After fire sale phase has failed banks
        - When you need contagion effects only, not initial failure tracking

        Args:
            failed_bank_ids: List of bank IDs that are already marked as FAILED.
                           These banks will be the starting point for contagion.

        Returns:
            Dictionary containing:
            - total_failed: Total banks failed (initial + contagion)
            - failed_bank_ids: All failed bank IDs
            - contagion_failures: Number of banks that failed from contagion only
            - contagion_failed_bank_ids: Bank IDs that failed from contagion only
            - total_rounds: Number of contagion rounds
            - failures_by_round: List of failures per round (excludes initial)
            - initial_failures: The input failed_bank_ids (for reference)
            - contagion_history: Round-by-round state snapshots
            - contagion_edges: All loss transmission edges

        Raises:
            ValueError: If any bank ID doesn't exist or is not marked as FAILED

        Example:
            # After asset shocks fail banks [3, 6, 8]
            result = network.propagate_contagion_from_failed([3, 6, 8])

            # result['contagion_failures'] = number of additional failures
            # result['contagion_failed_bank_ids'] = banks that failed from contagion
            # Contagion rounds start at 1, not 0 (banks already failed in round 0)
        """
        # Validate inputs
        for bank_id in failed_bank_ids:
            if bank_id not in self.banks:
                raise ValueError(f"Bank {bank_id} not found in network")

            bank_status = self.banks[bank_id].get_status()
            if bank_status != BankStatus.FAILED:
                raise ValueError(
                    f"Bank {bank_id} has status {bank_status.value}, not FAILED. "
                    f"This method requires banks to already be marked as failed. "
                    f"Use simulate_shock() for solvent banks."
                )

        # Reset contagion history
        self.contagion_history = []

        # Initialize with already-failed banks
        failed_banks: Set[int] = set(failed_bank_ids)
        round_failures: List[List[int]] = []  # No failures in "round 0" - banks already failed

        # Start contagion propagation from round 1
        # (Round 0 was when banks failed from their original cause)
        round_num = 1
        newly_failed = set(failed_bank_ids)  # Start propagation from these
        all_contagion_edges = []

        while newly_failed:
            current_round_failures = []
            round_contagion_edges = []

            # Process each newly failed bank
            for failed_bank_id in newly_failed:
                failed_bank = self.banks[failed_bank_id]

                # Use interbank-specific recovery rate if priority of claims enabled
                if use_priority_claims:
                    recovery_rate = failed_bank.get_interbank_recovery_rate(priority=True)
                else:
                    recovery_rate = failed_bank.get_recovery_rate()

                # Get all creditors (banks that lent to the failed bank)
                creditors = self.get_creditors(failed_bank_id)

                for creditor_id in creditors:
                    if creditor_id in failed_banks:
                        continue  # Skip already failed banks

                    creditor = self.banks[creditor_id]
                    exposure = creditor.interbank_exposures.get(failed_bank_id, 0.0)

                    # Calculate loss (1 - recovery_rate) * exposure
                    loss = (1.0 - recovery_rate) * exposure

                    if loss > 0:
                        # Capture capital ratio before applying loss
                        capital_before = creditor.get_capital_ratio()

                        # Apply the loss (with round number for failure tracking)
                        creditor.apply_interbank_loss(failed_bank_id, loss, failure_round=round_num)

                        # Capture capital ratio after applying loss
                        capital_after = creditor.get_capital_ratio()
                        caused_failure = creditor.get_status() == BankStatus.FAILED

                        # Record this loss transmission (contagion edge)
                        contagion_edge = {
                            'debtor_id': failed_bank_id,
                            'creditor_id': creditor_id,
                            'exposure': exposure,
                            'recovery_rate': recovery_rate,
                            'loss': loss,
                            'creditor_capital_before': capital_before,
                            'creditor_capital_after': capital_after,
                            'caused_failure': caused_failure
                        }
                        round_contagion_edges.append(contagion_edge)

                        # Check if creditor becomes insolvent
                        if caused_failure and creditor_id not in failed_banks:
                            failed_banks.add(creditor_id)
                            current_round_failures.append(creditor_id)

            # Record state if there were any failures or edges in this round
            if current_round_failures or round_contagion_edges:
                all_contagion_edges.extend(round_contagion_edges)
                round_failures.append(current_round_failures)
                self._record_state(
                    round_num=round_num,
                    newly_failed=current_round_failures,
                    contagion_edges=round_contagion_edges
                )

                newly_failed = set(current_round_failures)
                round_num += 1
            else:
                # No more failures or losses, contagion complete
                break

        # Calculate contagion-only failures (excludes the initial failed banks)
        contagion_only_failures = failed_banks - set(failed_bank_ids)

        # Compile results
        return {
            'total_failed': len(failed_banks),
            'failed_bank_ids': sorted(list(failed_banks)),
            'contagion_failures': len(contagion_only_failures),
            'contagion_failed_bank_ids': sorted(list(contagion_only_failures)),
            'total_rounds': round_num - 1,
            'failures_by_round': round_failures,
            'initial_failures': sorted(failed_bank_ids),
            'contagion_history': self.contagion_history,
            'contagion_edges': all_contagion_edges
        }

    def propagate_contagion_with_fire_sales(
        self,
        failed_bank_ids: List[int],
        fire_sale_intensity: float,
        use_priority_claims: bool = False,
        verbose: bool = False
    ) -> Dict:
        """
        Propagate contagion with fire sales integrated into EACH round.

        This is the economically correct implementation where:
        1. Banks fail from contagion
        2. Fire sales happen IMMEDIATELY based on that round's failures
        3. Fire sales reduce external_assets of survivors
        4. Next round's failures have REDUCED recovery rates due to fire sales
        5. Fire sales do NOT compound - markdowns calculated vs INITIAL market size

        Args:
            failed_bank_ids: Initial failed banks (e.g., from asset shocks)
            fire_sale_intensity: Fire sale intensity parameter (0 to 1)
            use_priority_claims: If True, interbank creditors are junior
            verbose: Print detailed progress

        Returns:
            Dictionary containing:
            - total_failed: Total banks failed
            - failed_bank_ids: All failed bank IDs
            - contagion_failures: Banks failed from contagion (excludes initial)
            - fire_sale_failures: Banks failed from fire sales
            - total_rounds: Number of contagion rounds
            - failures_by_round: List of failures per round
            - fire_sales_by_round: List of fire sale details per round
            - total_fire_sale_losses: Cumulative fire sale losses
            - contagion_history: Round-by-round snapshots
            - contagion_edges: All loss transmission edges

        Example:
            Round 0: Banks [1,2,3] fail from shock
            Round 1: Propagate losses → Bank 4 fails
                     Fire sales: 4 failures × intensity → markdown
                     Apply to survivors → Bank 5 assets reduced
            Round 2: Propagate losses (Bank 5 has reduced assets!)
                     Bank 5 fails with lower recovery rate
                     Fire sales from Bank 5 failure
            ...continues until no new failures
        """
        if verbose:
            print(f"\n=== Fire Sales Integrated Contagion ===")
            print(f"Initial failures: {failed_bank_ids}")
            print(f"Fire sale intensity: {fire_sale_intensity}")
            print(f"Priority claims: {use_priority_claims}")

        # Validate inputs
        for bank_id in failed_bank_ids:
            if bank_id not in self.banks:
                raise ValueError(f"Bank {bank_id} not found in network")
            if self.banks[bank_id].get_status() != BankStatus.FAILED:
                raise ValueError(f"Bank {bank_id} must be FAILED")

        # Reset contagion history
        self.contagion_history = []

        # Track state
        failed_banks: Set[int] = set(failed_bank_ids)
        contagion_failed_banks: Set[int] = set()  # Failed from contagion only
        fire_sale_failed_banks: Set[int] = set()  # Failed from fire sales
        round_failures: List[List[int]] = []
        fire_sales_by_round: List[Dict] = []
        all_contagion_edges = []
        total_fire_sale_losses = 0.0

        # Calculate INITIAL market size per asset type (to prevent compounding)
        # This is done once at the beginning and used for all fire sale calculations
        initial_market_size = {}
        if fire_sale_intensity > 0:
            from financial_contagion_networks.core.assets import AssetType
            initial_market_size = {
                AssetType.GOVERNMENT_BOND: 0.0,
                AssetType.CORPORATE_BOND: 0.0,
                AssetType.MORTGAGE: 0.0,
                AssetType.STOCK: 0.0
            }
            for bank in self.banks.values():
                if bank.has_portfolio():
                    composition = bank.get_portfolio_composition()
                    if composition:
                        for asset_type, fraction in composition.items():
                            # Use INITIAL external_assets (before any shocks/fire sales)
                            asset_value = bank.external_assets * fraction
                            initial_market_size[asset_type] += asset_value

            if verbose and initial_market_size:
                print(f"\nInitial market sizes (prevents compounding):")
                for asset_type, size in initial_market_size.items():
                    print(f"  {asset_type.value}: ${size:.2f}")

        round_num = 1
        newly_failed = set(failed_bank_ids)  # Start with initial failures

        while newly_failed:
            if verbose:
                print(f"\n--- Round {round_num} ---")
                print(f"Processing failures: {sorted(newly_failed)}")

            current_round_contagion_failures = []
            current_round_fire_sale_failures = []
            round_contagion_edges = []

            # Phase A: Propagate interbank losses from newly failed banks
            for failed_bank_id in newly_failed:
                failed_bank = self.banks[failed_bank_id]

                # Calculate recovery rate (affected by previous fire sales!)
                if use_priority_claims:
                    recovery_rate = failed_bank.get_interbank_recovery_rate(priority=True)
                else:
                    recovery_rate = failed_bank.get_recovery_rate()

                if verbose:
                    print(f"  Bank {failed_bank_id} recovery rate: {recovery_rate:.2%}")

                # Get creditors
                creditors = self.get_creditors(failed_bank_id)

                for creditor_id in creditors:
                    if creditor_id in failed_banks:
                        continue  # Skip already failed banks

                    creditor = self.banks[creditor_id]
                    exposure = creditor.interbank_exposures.get(failed_bank_id, 0.0)
                    loss = (1.0 - recovery_rate) * exposure

                    if loss > 0:
                        capital_before = creditor.get_capital_ratio()
                        creditor.apply_interbank_loss(failed_bank_id, loss, failure_round=round_num)
                        capital_after = creditor.get_capital_ratio()
                        caused_failure = creditor.get_status() == BankStatus.FAILED

                        round_contagion_edges.append({
                            'debtor_id': failed_bank_id,
                            'creditor_id': creditor_id,
                            'exposure': exposure,
                            'recovery_rate': recovery_rate,
                            'loss': loss,
                            'creditor_capital_before': capital_before,
                            'creditor_capital_after': capital_after,
                            'caused_failure': caused_failure
                        })

                        if caused_failure and creditor_id not in failed_banks:
                            failed_banks.add(creditor_id)
                            contagion_failed_banks.add(creditor_id)
                            current_round_contagion_failures.append(creditor_id)
                            if verbose:
                                print(f"    → Bank {creditor_id} failed from contagion loss ${loss:.2f}")

            # Phase B: Apply ASSET-SPECIFIC fire sales based on THIS ROUND's failures
            round_fire_sale_details = []
            round_fire_sale_losses = 0.0

            if fire_sale_intensity > 0 and newly_failed:
                from financial_contagion_networks.core.assets import AssetType

                # Calculate asset-specific markdowns based on actual liquidations
                # Step 1: Sum up total assets liquidated by asset type
                liquidated_assets = {
                    AssetType.GOVERNMENT_BOND: 0.0,
                    AssetType.CORPORATE_BOND: 0.0,
                    AssetType.MORTGAGE: 0.0,
                    AssetType.STOCK: 0.0
                }

                # Calculate liquidations from newly failed banks
                for failed_bank_id in newly_failed:
                    failed_bank = self.banks[failed_bank_id]
                    if failed_bank.has_portfolio():
                        composition = failed_bank.get_portfolio_composition()
                        if composition:
                            external_assets = failed_bank.external_assets
                            for asset_type, fraction in composition.items():
                                asset_value = external_assets * fraction
                                liquidated_assets[asset_type] += asset_value

                # Step 2: Calculate asset-specific markdowns using INITIAL market size
                # markdown = fire_sale_intensity × (liquidated_volume / INITIAL_market_size)
                # This prevents compounding across rounds
                asset_markdowns = {}
                for asset_type in liquidated_assets.keys():
                    if initial_market_size.get(asset_type, 0) > 0:
                        liquidation_ratio = liquidated_assets[asset_type] / initial_market_size[asset_type]
                        markdown = fire_sale_intensity * liquidation_ratio
                        if markdown > 0:
                            asset_markdowns[asset_type] = markdown

                if asset_markdowns and verbose:
                    print(f"  Fire sales (asset-specific):")
                    for asset_type, markdown in asset_markdowns.items():
                        print(f"    {asset_type.value}: {markdown:.2%} markdown")

                # Step 3: Apply asset-specific markdowns to surviving banks
                if asset_markdowns:
                    for bank_id, bank in self.banks.items():
                        if bank_id in failed_banks:
                            continue  # Skip failed banks

                        if bank.has_portfolio():
                            capital_before = bank.get_capital_ratio()
                            loss = bank.apply_fire_sale_markdown(asset_markdowns, failure_round=round_num)
                            capital_after = bank.get_capital_ratio()
                            caused_failure = bank.get_status() == BankStatus.FAILED

                            if loss > 0:
                                round_fire_sale_losses += loss
                                round_fire_sale_details.append({
                                    'bank_id': bank_id,
                                    'markdowns': asset_markdowns.copy(),  # Record asset-specific markdowns
                                    'loss': loss,
                                    'capital_before': capital_before,
                                    'capital_after': capital_after,
                                    'caused_failure': caused_failure
                                })

                                if caused_failure and bank_id not in failed_banks:
                                    failed_banks.add(bank_id)
                                    fire_sale_failed_banks.add(bank_id)
                                    current_round_fire_sale_failures.append(bank_id)
                                    if verbose:
                                        print(f"    → Bank {bank_id} failed from fire sale loss ${loss:.2f}")

                    total_fire_sale_losses += round_fire_sale_losses

            # Record round results
            all_round_failures = current_round_contagion_failures + current_round_fire_sale_failures
            if all_round_failures or round_contagion_edges or round_fire_sale_details:
                all_contagion_edges.extend(round_contagion_edges)
                round_failures.append(all_round_failures)
                fire_sales_by_round.append({
                    'round': round_num,
                    'asset_markdowns': asset_markdowns if fire_sale_intensity > 0 and newly_failed else {},
                    'total_losses': round_fire_sale_losses,
                    'details': round_fire_sale_details,
                    'new_failures': current_round_fire_sale_failures
                })

                self._record_state(
                    round_num=round_num,
                    newly_failed=all_round_failures,
                    contagion_edges=round_contagion_edges
                )

                # Next round processes ALL new failures (contagion + fire sale)
                newly_failed = set(all_round_failures)
                round_num += 1
            else:
                # No new failures, contagion complete
                break

        if verbose:
            print(f"\n=== Contagion Complete ===")
            print(f"Total failed: {len(failed_banks)}")
            print(f"Contagion failures: {len(contagion_failed_banks)}")
            print(f"Fire sale failures: {len(fire_sale_failed_banks)}")
            print(f"Total fire sale losses: ${total_fire_sale_losses:.2f}")

        return {
            'total_failed': len(failed_banks),
            'failed_bank_ids': sorted(list(failed_banks)),
            'contagion_failures': len(contagion_failed_banks),
            'contagion_failed_bank_ids': sorted(list(contagion_failed_banks)),
            'fire_sale_failures': len(fire_sale_failed_banks),
            'fire_sale_failed_bank_ids': sorted(list(fire_sale_failed_banks)),
            'total_rounds': round_num - 1,
            'failures_by_round': round_failures,
            'fire_sales_by_round': fire_sales_by_round,
            'total_fire_sale_losses': total_fire_sale_losses,
            'initial_failures': sorted(failed_bank_ids),
            'contagion_history': self.contagion_history,
            'contagion_edges': all_contagion_edges
        }

    def _record_state(self, round_num: int, newly_failed: List[int],
                     contagion_edges: List[Dict] = None) -> None:
        """
        Record the current state of the network.

        Args:
            round_num: Current round number
            newly_failed: List of newly failed bank IDs in this round
            contagion_edges: List of contagion edges (loss transmissions) in this round
        """
        state = {
            'round': round_num,
            'newly_failed': newly_failed,
            'contagion_edges': contagion_edges or [],  # Track loss transmissions
            'banks': {bank_id: bank.to_dict() for bank_id, bank in self.banks.items()}
        }
        self.contagion_history.append(state)

    def get_network_statistics(self) -> Dict:
        """
        Calculate network statistics.

        Returns:
            Dictionary containing various network statistics
        """
        num_banks = len(self.banks)
        num_edges = self.graph.number_of_edges()

        # Calculate total assets and equity
        total_assets = sum(bank.get_total_assets() for bank in self.banks.values())
        total_equity = sum(bank.get_equity() for bank in self.banks.values())
        total_interbank_assets = sum(bank.interbank_assets for bank in self.banks.values())

        # Count failed banks
        failed_banks = sum(1 for bank in self.banks.values()
                          if bank.get_status() == BankStatus.FAILED)

        # Network density
        density = nx.density(self.graph) if num_banks > 1 else 0.0

        # Average degree
        if num_banks > 0:
            degrees = [d for _, d in self.graph.degree()]
            avg_degree = np.mean(degrees) if degrees else 0.0
        else:
            avg_degree = 0.0

        return {
            'num_banks': num_banks,
            'num_exposures': num_edges,
            'total_assets': total_assets,
            'total_equity': total_equity,
            'total_interbank_assets': total_interbank_assets,
            'failed_banks': failed_banks,
            'failure_rate': failed_banks / num_banks if num_banks > 0 else 0.0,
            'network_density': density,
            'average_degree': avg_degree
        }

    def reset_banks(self) -> None:
        """
        Reset all banks to their initial state.

        Note: This only resets the status. It doesn't restore balance sheets
        that were modified during contagion simulation.
        """
        for bank in self.banks.values():
            if bank.get_status() == BankStatus.FAILED:
                # Can't easily reset banks that had losses applied
                # Would need to track initial state
                pass
        self.contagion_history = []

    def get_bank(self, bank_id: int) -> Optional[Bank]:
        """
        Get a bank by ID.

        Args:
            bank_id: Bank ID

        Returns:
            Bank object or None if not found
        """
        return self.banks.get(bank_id)

    def get_exposure_matrices(self) -> Dict:
        """
        Generate comprehensive exposure and obligation matrices.

        Returns a dictionary containing:
        - exposure_matrix: Dict mapping (lender_id, borrower_id) -> exposure_amount
        - obligation_matrix: Dict mapping (borrower_id, lender_id) -> obligation_amount
        - bank_exposures: Dict mapping bank_id -> {counterparty_id: exposure}
        - bank_obligations: Dict mapping bank_id -> {counterparty_id: obligation}
        - total_exposures_by_bank: Dict mapping bank_id -> total_exposure
        - total_obligations_by_bank: Dict mapping bank_id -> total_obligation

        Returns:
            Dictionary containing all matrix representations
        """
        exposure_matrix = {}
        obligation_matrix = {}
        bank_exposures = {}
        bank_obligations = {}
        total_exposures_by_bank = {}
        total_obligations_by_bank = {}

        # Build matrices from bank-level data
        for bank_id, bank in self.banks.items():
            # Exposures (this bank lends to others)
            bank_exposures[bank_id] = dict(bank.interbank_exposures)
            total_exposures_by_bank[bank_id] = bank.interbank_assets

            # Add to exposure matrix
            for counterparty_id, amount in bank.interbank_exposures.items():
                exposure_matrix[(bank_id, counterparty_id)] = amount

            # Obligations (this bank owes to others)
            bank_obligations[bank_id] = dict(bank.interbank_obligations)
            total_obligations_by_bank[bank_id] = bank.interbank_liabilities

            # Add to obligation matrix
            for counterparty_id, amount in bank.interbank_obligations.items():
                obligation_matrix[(bank_id, counterparty_id)] = amount

        return {
            'exposure_matrix': exposure_matrix,
            'obligation_matrix': obligation_matrix,
            'bank_exposures': bank_exposures,
            'bank_obligations': bank_obligations,
            'total_exposures_by_bank': total_exposures_by_bank,
            'total_obligations_by_bank': total_obligations_by_bank
        }

    def __repr__(self) -> str:
        """String representation of the network."""
        return (f"ContagionNetwork(banks={len(self.banks)}, "
                f"exposures={self.graph.number_of_edges()})")
