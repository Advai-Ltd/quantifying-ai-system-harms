#!/usr/bin/env python3
"""Test truncated normal distribution sampling."""

import numpy as np
import pytest
from financial_contagion_networks.config.models import DistributionConfig


def test_normal_distribution_respects_min_bound():
    """Test that normal distribution never samples below min."""
    dist = DistributionConfig(
        distribution='normal',
        mean=0.105,
        std=0.015,
        min=0.10,
        max=0.14
    )
    # Sample 1000 times, verify all >= 0.10
    rng = np.random.default_rng(42)
    samples = [dist.sample(rng) for _ in range(1000)]
    assert all(s >= 0.10 for s in samples), f"Some samples below min: {min(samples)}"
    print(f"✓ All 1000 samples >= 0.10 (min: {min(samples):.4f})")


def test_normal_distribution_respects_max_bound():
    """Test that normal distribution never samples above max."""
    dist = DistributionConfig(
        distribution='normal',
        mean=0.105,
        std=0.015,
        min=0.10,
        max=0.14
    )
    rng = np.random.default_rng(42)
    samples = [dist.sample(rng) for _ in range(1000)]
    assert all(s <= 0.14 for s in samples), f"Some samples above max: {max(samples)}"
    print(f"✓ All 1000 samples <= 0.14 (max: {max(samples):.4f})")


def test_normal_distribution_approximate_mean():
    """Test that truncated normal still approximates target mean."""
    dist = DistributionConfig(
        distribution='normal',
        mean=0.105,
        std=0.015,
        min=0.10,
        max=0.14
    )
    rng = np.random.default_rng(42)
    samples = [dist.sample(rng) for _ in range(10000)]
    mean = np.mean(samples)

    # Mean will be slightly higher than 0.105 due to truncation at 0.10
    # But should be close (within 0.005)
    assert 0.105 <= mean <= 0.110, f"Mean {mean:.4f} outside expected range [0.105, 0.110]"
    print(f"✓ Mean of 10000 samples: {mean:.4f} (target: 0.105)")


def test_uniform_distribution_still_works():
    """Test that uniform distribution is not affected by truncation changes."""
    dist = DistributionConfig(
        distribution='uniform',
        min=0.09,
        max=0.11
    )
    rng = np.random.default_rng(42)
    samples = [dist.sample(rng) for _ in range(1000)]

    assert all(0.09 <= s <= 0.11 for s in samples)
    mean = np.mean(samples)
    assert 0.095 <= mean <= 0.105  # Should be ~0.10
    print(f"✓ Uniform distribution still works (mean: {mean:.4f})")


def test_fixed_distribution_still_works():
    """Test that fixed distribution is not affected."""
    dist = DistributionConfig(
        distribution='fixed',
        value=0.12
    )
    rng = np.random.default_rng(42)
    samples = [dist.sample(rng) for _ in range(100)]

    assert all(s == 0.12 for s in samples)
    print("✓ Fixed distribution still works")


def test_normal_without_bounds_still_works():
    """Test that normal distribution works without min/max specified."""
    dist = DistributionConfig(
        distribution='normal',
        mean=0.105,
        std=0.015
    )
    rng = np.random.default_rng(42)
    samples = [dist.sample(rng) for _ in range(1000)]

    mean = np.mean(samples)
    # Without truncation, should be very close to target mean
    assert 0.100 <= mean <= 0.110, f"Mean {mean:.4f} outside expected range"
    print(f"✓ Normal without bounds works (mean: {mean:.4f})")


def test_normal_enforces_basel_iii_floor():
    """Test that Basel III 10% floor is enforced for periphery banks."""
    # Simulate periphery bank capital ratio sampling
    dist = DistributionConfig(
        distribution='normal',
        mean=0.105,  # 10.5% - just above regulatory minimum
        std=0.015,   # 1.5% std dev
        min=0.10,    # Basel III Tier 1 minimum (HARD REGULATORY FLOOR)
        max=0.14     # Well-capitalized periphery
    )

    rng = np.random.default_rng(42)
    samples = [dist.sample(rng) for _ in range(10000)]

    # Critical test: No bank should have capital ratio < 10%
    assert all(s >= 0.10 for s in samples), f"Some banks below Basel III floor: {min(samples)}"

    # Distribution should be reasonable
    mean = np.mean(samples)
    std = np.std(samples)

    print(f"✓ Basel III 10% floor enforced")
    print(f"  Sampled {len(samples)} banks:")
    print(f"    Min capital ratio: {min(samples)*100:.2f}%")
    print(f"    Max capital ratio: {max(samples)*100:.2f}%")
    print(f"    Mean capital ratio: {mean*100:.2f}%")
    print(f"    Std dev: {std*100:.2f}%")

    # Verify heterogeneity (std dev should be close to target)
    assert 0.008 <= std <= 0.018, f"Std dev {std:.4f} outside reasonable range"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
