# Post-Extension Fire Sale Parameter Sweep

## Overview

This experiment sweeps fire sale intensity from 0% to 15% to measure how fire sale market impact affects Basel III reform effectiveness, using the **fixed fire sale mechanics** (no compounding, asset-specific markdowns).

## Research Question

With realistic fire sale implementation (no compounding, 0.05 baseline intensity), how does varying fire sale intensity affect the systemic crisis threshold in the post-2008 reformed system?

## Key Differences from Pre-Extension

**Fire Sale Mechanics:**
- ✓ No compounding (uses initial market size)
- ✓ Asset-specific markdowns
- ✓ Heterogeneous portfolios (Option F)
- ✓ Variable interbank exposure (Option A)
- ✓ Stochastic topology (Option D)

**Intensity Range:**
- Pre-extension: 0.15 to 1.50 (15% to 150%)
- **Post-extension: 0.00 to 0.15 (0% to 15%)** - More sensitive range

## Fire Sale Intensity Levels

Testing 8 intensity levels:
- **0.00 (0%)**: No fire sales (counterfactual)
- **0.01 (1%)**: Minimal market impact
- **0.03 (3%)**: Low market impact
- **0.05 (5%)**: Moderate impact (current baseline from post_2008)
- **0.07 (7%)**: Elevated market stress
- **0.10 (10%)**: High market stress
- **0.12 (12%)**: Severe market stress
- **0.15 (15%)**: Extreme stress (pre-extension baseline)

## Baseline Results

From `experiments/post_extension/post_2008/`:
- Fire sale intensity: 0.05 (5%)
- Systemic crisis threshold: ~-14% shock
- Reform benefit vs pre-2008: +9pp improvement

## Expected Results

We expect to see:
1. **0% intensity**: Highest resilience (no fire sale amplification)
2. **1-3% intensity**: Minimal threshold erosion
3. **5% intensity**: Current baseline (should match post_2008 results)
4. **7-10% intensity**: Moderate threshold erosion (2-4pp lost)
5. **12-15% intensity**: Significant erosion (4-6pp lost)

## Computational Requirements

- **Total simulations**: 8 intensities × 29 shocks × 200 runs = **46,400 simulations**
- **Estimated runtime**: ~15-20 hours sequential, ~5-7 hours with parallelization
- **Storage**: ~1.5 GB

## Usage

### Full Experiment (Recommended)

```bash
uv run python experiments/post_extension/fire_sale_sweep/run.py --yes
```

This runs all 8 fire sale intensities (0% to 15%) with shocks from -1% to -30%.

### Quick Test (2 intensities, 10 shocks, 50 runs = 1,000 simulations)

```bash
uv run python experiments/post_extension/fire_sale_sweep/run.py \
  --intensities 0.00 0.05 \
  --min-shock -0.05 --max-shock -0.15 --step 0.01 \
  --num-runs 50 \
  --yes
```

### Custom Intensity Range

```bash
uv run python experiments/post_extension/fire_sale_sweep/run.py \
  --intensities 0.00 0.03 0.05 0.10 0.15 \
  --yes
```

## Output Structure

```
output/run_<TIMESTAMP>/
├── fire_sale_sweep_summary.json       # Complete results with thresholds
├── intensity_00pct/                   # No fire sales
│   ├── shock_01p0/
│   ├── shock_02p0/
│   └── ...
├── intensity_01pct/                   # 1% fire sales
├── intensity_03pct/                   # 3% fire sales
├── intensity_05pct/                   # 5% fire sales (baseline)
├── intensity_07pct/                   # 7% fire sales
├── intensity_10pct/                   # 10% fire sales
├── intensity_12pct/                   # 12% fire sales
└── intensity_15pct/                   # 15% fire sales (pre-extension baseline)
```

## Key Metrics

### Primary: Threshold Changes
- Systemic crisis threshold (>30% failures) at each intensity
- Change from 5% baseline (positive = more resilient, negative = less resilient)

### Secondary: Fire Sale Amplification
- Fire sale losses vs total losses
- Contagion propagation depth
- Systemic crisis probability curves

## Comparison to Pre-Extension

| Aspect | Pre-Extension | **Post-Extension** |
|--------|---------------|-------------------|
| **Intensity range** | 15% to 150% | **0% to 15%** |
| **Fire sale compounding** | Yes (buggy) | **No (fixed)** |
| **Heterogeneous portfolios** | No | **Yes (Option F)** |
| **Baseline intensity** | 15% | **5%** |
| **Sensitivity** | Coarse (high range) | **Fine (realistic range)** |

## Policy Implications

This experiment helps answer:
1. How much do fire sales contribute to systemic risk in Basel III systems?
2. What is the "safe" level of fire sale intensity before reform benefits erode?
3. Should regulators impose limits on synchronized selling (AI, algorithmic trading)?

## Prerequisites

Run `post_2008/` first to establish the 5% baseline:
```bash
cd ../post_2008/
uv run python run.py --yes
```

## Next Steps

After completing this sweep:
1. Generate comparison graphs showing threshold erosion
2. Calculate fire sale amplification ratios at each intensity
3. Compare to pre-extension results (15% baseline)
4. Document policy implications for AI-driven trading
