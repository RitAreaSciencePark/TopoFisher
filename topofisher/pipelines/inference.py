#!/usr/bin/env python3
"""
Inference script for trained TopoFisher pipelines.

Usage:
    python -m topofisher.pipelines.inference <checkpoint_path> [options]

Examples:
    python -m topofisher.pipelines.inference experiments/grf/results/topk_mlp/pipeline.pt
    python -m topofisher.pipelines.inference pipeline.pt --n-samples 1000 --seed 100
"""
import argparse
import torch
from pathlib import Path

from ..config import load_pipeline_checkpoint, AnalysisConfig


def run_inference(
    checkpoint_path: str,
    n_samples: int = 2000,
    seed_cov: int = 142,
    seed_ders: list = None,
    verbose: bool = True
):
    """
    Run Fisher information inference on a trained pipeline.

    Args:
        checkpoint_path: Path to pipeline.pt checkpoint
        n_samples: Number of samples for covariance and derivatives
        seed_cov: Seed for fiducial samples
        seed_ders: Seeds for derivative samples (auto-generated if None)
        verbose: Print detailed output

    Returns:
        FisherResult object
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load pipeline
    if verbose:
        print("=" * 60)
        print("Loading pipeline checkpoint")
        print("=" * 60)

    pipeline, config = load_pipeline_checkpoint(checkpoint_path)

    if verbose:
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Pipeline: {type(pipeline).__name__}")
        print(f"Simulator: {type(pipeline.simulator).__name__ if pipeline.simulator else 'None'}")
        print(f"Filtration: {type(pipeline.filtration).__name__ if pipeline.filtration else 'None'}")
        print(f"Vectorization: {type(pipeline.vectorization).__name__}")
        print(f"Compression: {type(pipeline.compression).__name__}")

        # Show TopK k values
        if hasattr(pipeline.vectorization, 'layers'):
            print(f"\nVectorization layers:")
            for i, layer in enumerate(pipeline.vectorization.layers):
                print(f"  Layer {i}: k={getattr(layer, 'k', 'N/A')}")

    # Create analysis config
    theta_fid = config.analysis.theta_fid
    delta_theta = config.analysis.delta_theta

    if seed_ders is None:
        seed_ders = [seed_cov + i + 1 for i in range(len(theta_fid))]

    analysis_config = AnalysisConfig(
        theta_fid=theta_fid,
        delta_theta=delta_theta,
        n_s=n_samples,
        n_d=n_samples,
        seed_cov=seed_cov,
        seed_ders=seed_ders,
        cache=None
    )

    if verbose:
        print(f"\n" + "=" * 60)
        print("Running inference")
        print("=" * 60)
        print(f"theta_fid: {theta_fid.tolist()}")
        print(f"delta_theta: {delta_theta.tolist()}")
        print(f"n_samples: {n_samples}")
        print(f"seeds: cov={seed_cov}, ders={seed_ders}")

    # Run inference
    pipeline.eval()
    with torch.no_grad():
        result = pipeline(analysis_config)

    if verbose:
        print(f"\n" + "=" * 60)
        print("Results")
        print("=" * 60)

        print(f"\nFisher Matrix:")
        print(result.fisher_matrix.cpu().numpy())

        print(f"\nlog|F| = {result.log_det_fisher.cpu().item():.4f}")

        print(f"\nConstraints (1σ):")
        for i, sigma in enumerate(result.constraints.cpu()):
            print(f"  θ_{i}: ±{sigma.item():.4f}")

        result.print_gaussianity()

        # Compare with theoretical if available
        if pipeline.simulator and hasattr(pipeline.simulator, 'theoretical_fisher_matrix'):
            print(f"\n" + "=" * 60)
            print("Comparison with Theoretical")
            print("=" * 60)

            F_theory = pipeline.simulator.theoretical_fisher_matrix(theta_fid)
            log_det_theory = torch.logdet(F_theory)
            constraints_theory = torch.sqrt(torch.diag(torch.linalg.inv(F_theory)))

            print(f"\n{'Method':<25} {'log|F|':>10} {'σ(θ_0)':>10} {'σ(θ_1)':>10}")
            print("-" * 58)
            print(f"{'Theoretical':<25} {log_det_theory.cpu().item():>10.2f} "
                  f"{constraints_theory[0].cpu().item():>10.4f} {constraints_theory[1].cpu().item():>10.4f}")
            print(f"{'Pipeline':<25} {result.log_det_fisher.cpu().item():>10.2f} "
                  f"{result.constraints[0].cpu().item():>10.4f} {result.constraints[1].cpu().item():>10.4f}")

            ratio = (result.log_det_fisher.cpu() / log_det_theory.cpu()).item()
            print(f"\nEfficiency: {ratio:.1%} of theoretical maximum")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Run Fisher information inference on trained pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m topofisher.pipelines.inference experiments/grf/results/topk_mlp/pipeline.pt
    python -m topofisher.pipelines.inference pipeline.pt --n-samples 1000
    python -m topofisher.pipelines.inference pipeline.pt --seed 100 --quiet
        """
    )
    parser.add_argument('checkpoint', type=str, help='Path to pipeline.pt checkpoint')
    parser.add_argument('--n-samples', type=int, default=500,
                        help='Number of samples for covariance and derivatives (default: 500)')
    parser.add_argument('--seed', type=int, default=142,
                        help='Base seed for reproducibility (default: 142)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress detailed output')

    args = parser.parse_args()

    result = run_inference(
        checkpoint_path=args.checkpoint,
        n_samples=args.n_samples,
        seed_cov=args.seed,
        verbose=not args.quiet
    )

    if not args.quiet:
        print("\n" + "=" * 60)
        print("Inference completed successfully!")
        print("=" * 60)


if __name__ == '__main__':
    main()
