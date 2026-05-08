"""
Gaussianity testing for Fisher analysis.
"""
from typing import Dict, List, Tuple
import torch
import numpy as np
from scipy import stats


def test_gaussianity(
    summaries: List[torch.Tensor],
    alpha: float = 0.05,
    mode: str = "summary",
    verbose: bool = True,
    n_test: int = 2000,
) -> Tuple[Dict, bool]:
    """
    Test Gaussianity of summary statistics using Kolmogorov-Smirnov test.

    Tests each feature in each dataset (fiducial + derivatives) for Gaussianity.

    Args:
        summaries: List of summary tensors [fiducial, theta_minus_0, theta_plus_0, ...]
            Each tensor has shape (n_samples, n_features)
        alpha: Significance level for hypothesis test (default 0.05)
        mode: Output mode - "summary" (compact per-feature view) or "detailed" (per-dataset view)
        verbose: If True, print results (default True)
        n_test: Maximum number of samples to use per dataset for the KS test
            (default 2000). Using all samples (e.g. 20000) makes the KS
            threshold ~0.0096 — so tight that trivial kurtosis deviations of
            ~0.1 are flagged even though they have no practical impact on the
            Fisher matrix. With n_test=2000 the threshold is ~0.030, which
            detects excess kurtosis ≳ 0.5 where the Gaussian approximation
            starts to degrade meaningfully.

    Returns:
        Tuple of (results_dict, all_gaussian_flag) where:
            - results_dict: Dictionary with test results per dataset and feature
            - all_gaussian_flag: True if more than half of datasets have all features Gaussian
    """
    # Generate dataset names
    n_datasets = len(summaries)
    n_params = (n_datasets - 1) // 2
    dataset_names = ['Fiducial']
    for i in range(n_params):
        dataset_names.extend([f'θ_{i}-', f'θ_{i}+'])

    # Subsample each dataset to n_test rows so the KS threshold stays
    # at a practically meaningful level regardless of total sample count.
    rng = np.random.default_rng(seed=0)
    summaries = [
        s[rng.choice(s.shape[0], size=min(n_test, s.shape[0]), replace=False)]
        if s.shape[0] > n_test else s
        for s in summaries
    ]

    # Run tests on all datasets
    results = {}

    for summaries_tensor, dataset_name in zip(summaries, dataset_names):
        n_features = summaries_tensor.shape[1]
        summaries_np = summaries_tensor.detach().cpu().numpy()

        dataset_results = {}
        dataset_all_gaussian = True

        for feature_idx in range(n_features):
            feature_data = summaries_np[:, feature_idx]

            # Standardize to mean=0, std=1
            mean = feature_data.mean()
            std = feature_data.std()

            if std < 1e-10:  # Handle zero-variance features
                is_gaussian = False
                ks_stat = np.nan
                ks_pvalue = 0.0
            else:
                standardized = (feature_data - mean) / std
                # KS test against standard normal
                ks_stat, ks_pvalue = stats.kstest(standardized, 'norm')
                is_gaussian = ks_pvalue > alpha

            dataset_results[f"feature_{feature_idx}"] = {
                'mean': mean,
                'std': std,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'is_gaussian': is_gaussian
            }

            if not is_gaussian:
                dataset_all_gaussian = False

        # Dataset summary
        n_gaussian = sum(1 for v in dataset_results.values() if v['is_gaussian'])
        n_total = len(dataset_results)

        results[dataset_name] = {
            'features': dataset_results,
            'all_gaussian': dataset_all_gaussian,
            'n_gaussian': n_gaussian,
            'n_total': n_total
        }

    # Compute overall pass: more than half of datasets have ALL features Gaussian
    n_datasets_all_gaussian = sum(1 for r in results.values() if r['all_gaussian'])
    overall_pass = n_datasets_all_gaussian > (n_datasets / 2)

    # Print results based on mode
    if verbose:
        if mode == "summary":
            # Summary mode: compact per-feature view
            print("\n" + "=" * 80)
            print("Gaussianity Test (Kolmogorov-Smirnov) - Summary View")
            print("=" * 80)

            n_features = results[dataset_names[0]]['n_total']
            print(f"\nPer-Feature Summary (across {n_datasets} datasets):")

            for feature_idx in range(n_features):
                # Check which datasets pass for this feature
                failing_datasets = []
                for dataset_name in dataset_names:
                    feature_key = f"feature_{feature_idx}"
                    if not results[dataset_name]['features'][feature_key]['is_gaussian']:
                        failing_datasets.append(dataset_name)

                n_passing = n_datasets - len(failing_datasets)
                status = '✓' if n_passing == n_datasets else ''

                if n_passing == n_datasets:
                    print(f"  Feature {feature_idx:3d}: {n_passing}/{n_datasets} Gaussian {status}")
                else:
                    failing_str = ', '.join(failing_datasets)
                    print(f"  Feature {feature_idx:3d}: {n_passing}/{n_datasets} Gaussian (fails: {failing_str})")

            # Overall summary
            total_tests = sum(r['n_total'] for r in results.values())
            total_passed = sum(r['n_gaussian'] for r in results.values())

            print("\n" + "=" * 80)
            if overall_pass:
                print(f"✓ PASS: More than half of datasets ({n_datasets_all_gaussian}/{n_datasets}) have all features Gaussian")
            else:
                print(f"✗ FAIL: Only {n_datasets_all_gaussian}/{n_datasets} datasets have all features Gaussian (need > {n_datasets/2:.0f})")
                
            print(f"Overall: {total_passed}/{total_tests} tests passed")
            print(f"Datasets with ALL features Gaussian: {n_datasets_all_gaussian}/{n_datasets}")
            print(f"Note: p-value > {alpha} suggests data is consistent with Gaussian")
            print("=" * 80)

        elif mode == "detailed":
            # Detailed mode: per-dataset breakdown
            print("\n" + "=" * 80)
            print("Gaussianity Test (Kolmogorov-Smirnov) - Detailed View")
            print("Testing all datasets: fiducial + derivatives")
            print("=" * 80)

            for summaries_tensor, dataset_name in zip(summaries, dataset_names):
                dataset_result = results[dataset_name]

                print(f"\n{'='*80}")
                print(f"Dataset: {dataset_name} (shape: {summaries_tensor.shape})")
                print(f"{'='*80}")

                # Show first 3 features + any failures
                for feature_idx in range(dataset_result['n_total']):
                    feature_key = f"feature_{feature_idx}"
                    feature_data = dataset_result['features'][feature_key]
                    is_gaussian = feature_data['is_gaussian']

                    if not is_gaussian or feature_idx < 3:
                        mean = feature_data['mean']
                        std = feature_data['std']
                        ks_stat = feature_data['ks_statistic']
                        ks_pvalue = feature_data['ks_pvalue']

                        if np.isnan(ks_stat):
                            print(f"  Feature {feature_idx:3d}: ZERO VARIANCE - skipped")
                        else:
                            status = '✓ Gaussian' if is_gaussian else '✗ Non-Gaussian'
                            print(f"  Feature {feature_idx:3d}: mean={mean:8.4f}, std={std:8.4f}, "
                                  f"KS={ks_stat:.4f}, p={ks_pvalue:.4f} {status}")

                n_gaussian = dataset_result['n_gaussian']
                n_total = dataset_result['n_total']
                dataset_all_gaussian = dataset_result['all_gaussian']

                print(f"  Summary: {n_gaussian}/{n_total} features Gaussian")
                status = '✓' if dataset_all_gaussian else '✗'
                print(f"  {status} {'All features Gaussian' if dataset_all_gaussian else 'Some features non-Gaussian'}")

            # Overall summary
            total_tests = sum(r['n_total'] for r in results.values())
            total_passed = sum(r['n_gaussian'] for r in results.values())

            print("\n" + "=" * 80)
            print("OVERALL SUMMARY")
            print("=" * 80)
            for dataset_name, dataset_result in results.items():
                status = '✓' if dataset_result['all_gaussian'] else '✗'
                print(f"  {status} {dataset_name:<15}: {dataset_result['n_gaussian']}/{dataset_result['n_total']} Gaussian")
            print("=" * 80)
            print(f"Total: {total_passed}/{total_tests} tests passed")
            print(f"Datasets with ALL features Gaussian: {n_datasets_all_gaussian}/{n_datasets}")
            print(f"Note: p-value > {alpha} suggests data is consistent with Gaussian")

            if overall_pass:
                print(f"✓ PASS: More than half of datasets ({n_datasets_all_gaussian}/{n_datasets}) have all features Gaussian")
            else:
                print(f"✗ FAIL: Only {n_datasets_all_gaussian}/{n_datasets} have all features Gaussian (need > {n_datasets/2:.0f})")
            print("=" * 80)

        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'summary' or 'detailed'.")

    return results, overall_pass
