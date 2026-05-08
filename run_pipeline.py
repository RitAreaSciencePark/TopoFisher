#!/usr/bin/env python3
"""
Run a TopoFisher pipeline from a YAML configuration file.

Usage:
    python run_pipeline.py config.yaml [--train] [--verbose]

Examples:
    # Run inference only
    python run_pipeline.py topofisher/examples/grf/topk_moped.yaml

    # Train learnable components
    python run_pipeline.py topofisher/examples/grf/cubical_mlp.yaml --train
"""
import argparse
import json
import sys
import random
from pathlib import Path
import torch
import numpy as np
import yaml
import pickle
from topofisher.config import load_pipeline_config, create_pipeline_from_config
from topofisher.config import create_simulator, create_filtration, SimulatorConfig, FiltrationConfig
from topofisher.pipelines import BasePipeline
from topofisher.pipelines.cached import CachedCompressionPipeline


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For full determinism (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def extract_expanded_config(config, pipeline):
    """Extract fully expanded configuration including auto-selected values."""
    from dataclasses import asdict
    from topofisher.vectorizations import CombinedVectorization, TopKLayer

    # Convert config to dict (handle None for optional components)
    expanded = {
        'experiment': asdict(config.experiment),
        'analysis': asdict(config.analysis),
        'vectorization': asdict(config.vectorization),
        'compression': asdict(config.compression),
    }

    if config.simulator:
        expanded['simulator'] = asdict(config.simulator)
    if config.filtration:
        expanded['filtration'] = asdict(config.filtration)
    if config.training:
        expanded['training'] = asdict(config.training)

    # Extract auto-selected k values from TopK layers
    if isinstance(pipeline.vectorization, CombinedVectorization):
        layers = []
        for layer in pipeline.vectorization.layers:
            if isinstance(layer, TopKLayer) and layer.k is not None:
                layers.append({
                    'type': 'topk',
                    'params': {'k': layer.k}  # Include auto-selected k
                })
        if layers:
            expanded['vectorization']['params']['layers'] = layers

    return expanded


def _restore_fitted_state_from_checkpoint(pipeline):
    """
    After load_state_dict, restore Python-only 'fitted' flags on vectorization layers.

    state_dict carries the buffer values (input_mean, input_std, pers_threshold for
    PersLayLayer; grid / log_temp_min for DiffCurves) but not the plain-Python
    self.fitted attribute set in finalize_fit(). Detect fitted state from the
    buffers themselves and re-set the flag so forward() doesn't raise.
    """
    try:
        from topofisher.vectorizations.perslay import PersLayLayer
        has_perslay = True
    except ImportError:
        has_perslay = False

    try:
        from topofisher.vectorizations.differentiable_persistence_curves import (
            DifferentiablePersistenceCurveLayer,
        )
        has_diffcurves = True
    except ImportError:
        has_diffcurves = False

    vec = getattr(pipeline, 'vectorization', None)
    if vec is None:
        return
    layers = getattr(vec, 'layers', [vec])
    any_restored = False
    for layer in layers:
        if has_perslay and isinstance(layer, PersLayLayer):
            # pers_threshold defaults to +inf; if it's finite, finalize_fit() was called.
            # input_std defaults to all-ones; non-unit std also signals a completed fit.
            pers_thr = layer.pers_threshold.item()
            std_non_unit = not torch.allclose(layer.input_std, torch.ones_like(layer.input_std))
            if pers_thr < float('inf') or std_non_unit:
                layer.fitted = True
                any_restored = True
                print(f"    Restored PersLayLayer fitted state "
                      f"(pers_threshold={pers_thr:.4g}, input_std={layer.input_std.tolist()})")
        if has_diffcurves and isinstance(layer, DifferentiablePersistenceCurveLayer):
            grid = layer.grid
            if grid.abs().sum() > 0:
                layer.fitted = True
                any_restored = True
                if not layer.learn_temperature:
                    grid_step = (grid[-1] - grid[0]).item() / max(len(grid) - 1, 1)
                    if hasattr(layer, '_user_temperature') and layer._user_temperature is not None:
                        layer._temperature = layer._user_temperature
                    else:
                        layer._temperature = grid_step * layer.tau_scale
                print(f"    Restored DiffCurves fitted state: "
                      f"grid [{grid[0]:.4f}, {grid[-1]:.4f}]")

    # Prevent pipelines from re-fitting the vectorization (which would overwrite
    # the checkpoint's normalization buffers and break warm-start consistency).
    # _vectorization_fitted is checked in LearnableFiltrationPipeline and
    # LearnableBasePipeline.train_component before the "Pre-fitting" phase.
    if any_restored:
        pipeline._vectorization_fitted = True
        print("    Skipping vectorization re-fit (checkpoint normalization preserved)")


def main():
    parser = argparse.ArgumentParser(description='Run TopoFisher pipeline from YAML config')
    parser.add_argument('config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--train', action='store_true', help='Train learnable components')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--save-results', action='store_true', default=True,
                        help='Save results to output directory')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate from config')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory from config')
    parser.add_argument('--lambda-k', type=float, default=None,
                        help='Override kurtosis regularization strength')
    parser.add_argument('--lambda-s', type=float, default=None,
                        help='Override skewness regularization strength')
    parser.add_argument('--n-epochs', type=int, default=None,
                        help='Override number of training epochs')
    parser.add_argument('--freeze-cnn-epochs', type=int, default=None,
                        help='Override freeze_cnn_epochs (0 to skip CNN freeze, e.g. for warm-start extensions)')
    parser.add_argument('--lr-warmup', type=int, default=None,
                        help='Override lr_warmup epochs (0 to disable warmup, e.g. for warm-start extensions)')
    parser.add_argument('--lr-scheduler', type=str, default=None,
                        help='Override lr_scheduler ("plateau", "cosine", "cosine_restarts", "none")')
    parser.add_argument('--lr-T0', type=int, default=None,
                        help='Override lr_T0 (cosine restart period in epochs)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch_size (larger = lower gradient noise, same epoch compute)')
    parser.add_argument('--alpha-k', type=int, default=None,
                        help='Override k (nearest neighbors) for filtration')
    parser.add_argument('--hidden-dims', type=str, default=None,
                        help='Override hidden_dims for filtration (JSON list, e.g., "[20, 10]")')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override random seed for training')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override data_dir: load/save pre-generated simulation data from this path')
    parser.add_argument('--seed-cov', type=int, default=None,
                        help='Override seed_cov for covariance estimation (must match pre-generated data metadata)')
    parser.add_argument('--seed-ders', type=str, default=None,
                        help='Override seed_ders for derivative estimation (JSON list, e.g., "[143, 144]")')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Load pretrained pipeline weights before inference')

    args = parser.parse_args()

    # Load configuration
    print(f"\n{'='*60}")
    print(f"Loading configuration from: {args.config}")
    print(f"{'='*60}")

    try:
        config = load_pipeline_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Apply command line overrides
    if args.lr is not None and config.training is not None:
        print(f"Overriding learning rate: {config.training.lr} → {args.lr}")
        config.training.lr = args.lr

    if args.output_dir is not None:
        print(f"Overriding output directory: {config.experiment.output_dir} → {args.output_dir}")
        config.experiment.output_dir = args.output_dir

    if args.lambda_k is not None and config.training is not None:
        print(f"Overriding lambda_k: {config.training.lambda_k} → {args.lambda_k}")
        config.training.lambda_k = args.lambda_k

    if args.lambda_s is not None and config.training is not None:
        print(f"Overriding lambda_s: {config.training.lambda_s} → {args.lambda_s}")
        config.training.lambda_s = args.lambda_s

    if args.n_epochs is not None and config.training is not None:
        print(f"Overriding n_epochs: {config.training.n_epochs} → {args.n_epochs}")
        config.training.n_epochs = args.n_epochs

    if args.freeze_cnn_epochs is not None and config.training is not None:
        print(f"Overriding freeze_cnn_epochs: {config.training.freeze_cnn_epochs} → {args.freeze_cnn_epochs}")
        config.training.freeze_cnn_epochs = args.freeze_cnn_epochs

    if args.lr_warmup is not None and config.training is not None:
        print(f"Overriding lr_warmup: {config.training.lr_warmup} → {args.lr_warmup}")
        config.training.lr_warmup = args.lr_warmup

    if args.lr_scheduler is not None and config.training is not None:
        print(f"Overriding lr_scheduler: {config.training.lr_scheduler} → {args.lr_scheduler}")
        config.training.lr_scheduler = args.lr_scheduler

    if args.lr_T0 is not None and config.training is not None:
        print(f"Overriding lr_T0: {config.training.lr_T0} → {args.lr_T0}")
        config.training.lr_T0 = args.lr_T0

    if args.batch_size is not None and config.training is not None:
        print(f"Overriding batch_size: {config.training.batch_size} → {args.batch_size}")
        config.training.batch_size = args.batch_size

    if args.seed is not None and config.training is not None:
        print(f"Overriding seed: {config.training.seed} → {args.seed}")
        config.training.seed = args.seed

    # Filtration parameter overrides
    if args.alpha_k is not None and config.filtration is not None:
        print(f"Overriding filtration k: {config.filtration.params.get('k', 'N/A')} → {args.alpha_k}")
        config.filtration.params['k'] = args.alpha_k

    if args.hidden_dims is not None and config.filtration is not None:
        hidden_dims = json.loads(args.hidden_dims)
        print(f"Overriding filtration hidden_dims: {config.filtration.params.get('hidden_dims', 'N/A')} → {hidden_dims}")
        config.filtration.params['hidden_dims'] = hidden_dims

    # Data directory override (pre-generated simulation data)
    if args.data_dir is not None:
        print(f"Overriding data_dir: {config.analysis.data_dir} → {args.data_dir}")
        config.analysis.data_dir = args.data_dir

    # Analysis seed overrides (must match pre-generated data metadata)
    if args.seed_cov is not None:
        print(f"Overriding seed_cov: {config.analysis.seed_cov} → {args.seed_cov}")
        config.analysis.seed_cov = args.seed_cov

    if args.seed_ders is not None:
        seed_ders = json.loads(args.seed_ders)
        print(f"Overriding seed_ders: {list(config.analysis.seed_ders)} → {seed_ders}")
        config.analysis.seed_ders = seed_ders

    # Set random seeds for reproducibility
    if config.training is not None:
        print(f"\nSetting random seed: {config.training.seed}")
        set_seed(config.training.seed)

    # Print configuration summary
    print(f"\nExperiment: {config.experiment.name}")
    if config.experiment.description:
        print(f"Description: {config.experiment.description}")
    print(f"Output directory: {config.experiment.output_dir}")
    print(f"\nAnalysis parameters:")
    print(f"  theta_fid: {config.analysis.theta_fid}")
    print(f"  delta_theta: {config.analysis.delta_theta}")
    print(f"  n_s: {config.analysis.n_s}, n_d: {config.analysis.n_d}")
    if config.analysis.data_dir:
        print(f"  data_dir: {config.analysis.data_dir}")
    print(f"\nPipeline components:")
    print(f"  Simulator: {config.simulator.type if config.simulator else 'N/A (pre-generated data)'}")
    print(f"  Filtration: {config.filtration.type if config.filtration else 'N/A (cached)'}")
    print(f"  Vectorization: {config.vectorization.type} (trainable={config.vectorization.trainable})")
    print(f"  Compression: {config.compression.type} (trainable={config.compression.trainable})")

    # Create pipeline
    print(f"\n{'='*60}")
    print("Creating pipeline...")
    print(f"{'='*60}")

    try:
        pipeline, config = create_pipeline_from_config(config)
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        sys.exit(1)

    # Load pretrained checkpoint if requested (transfer learning / frozen inference)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        pipeline.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {args.checkpoint}")
        _restore_fitted_state_from_checkpoint(pipeline)
        # load_state_dict with map_location='cpu' may replace GPU parameters
        # with CPU tensors; re-move the pipeline to its target device.
        pipeline.to(pipeline.device)

    # Use analysis config directly (no conversion needed)
    analysis_config = config.analysis

    # Check for generate mode (special case - no training, just save data)
    if analysis_config.cache is not None and analysis_config.cache.mode == "generate":
        print(f"\n{'='*60}")
        print("Running in GENERATE mode")
        print(f"{'='*60}")

        result_dict = pipeline.run(analysis_config)

        print(f"\nGeneration complete!")
        print(f"  Data type: {result_dict.get('data_type', 'unknown')}")
        if 'saved_diagrams' in result_dict:
            print(f"  Saved to: {result_dict['saved_diagrams']}")
        if 'saved_summaries' in result_dict:
            print(f"  Saved to: {result_dict['saved_summaries']}")

        print(f"\n{'='*60}")
        print("Generation completed successfully!")
        print(f"{'='*60}\n")
        return

    # Training mode (cache load or regular trainable)
    needs_training = (analysis_config.cache is not None and analysis_config.cache.mode == "load") or \
                     (config.is_trainable() and args.train)

    if needs_training:
        if not config.training:
            print("Error: Training config required")
            sys.exit(1)

        print(f"\nTraining {config.get_trainable_component()} component...")
        print(f"Training config:")
        print(f"  Epochs: {config.training.n_epochs}")
        print(f"  Learning rate: {config.training.lr}")
        print(f"  Batch size: {config.training.batch_size}")

        # Generate/load data
        print("\nPreparing data...")
        data = pipeline.generate_data(analysis_config)

        # Train the pipeline (run() will free `data` after split_data)
        training_result = pipeline.run(
            config=analysis_config,
            training_config=config.training,
            data=data
        )
        del data  # drop local ref so gc can reclaim
        result = training_result['test_result']
        training_history = training_result.get('training_history', None)

    elif config.is_trainable() and not args.train:
        print("\nWarning: Pipeline has trainable components but --train not specified")
        print("Running inference only (may fail if components aren't pre-trained)")
        result = pipeline(analysis_config)
        training_history = None

    else:
        # Run inference
        print("\nRunning pipeline...")
        result = pipeline(analysis_config)
        training_history = None

    # Print results
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")

    print(f"\nFisher Information Matrix:")
    print(result.fisher_matrix.detach().cpu().numpy())

    print(f"\nlog|F| = {result.log_det_fisher.cpu().item():.4f}")

    print(f"\nParameter constraints (1σ):")
    for i, (param, sigma) in enumerate(zip(config.analysis.theta_fid, result.constraints.cpu())):
        print(f"  θ_{i} = {param:.2f} ± {sigma.item():.4f}")

    # Gaussianity check
    result.print_gaussianity()

    # Compare with theoretical if available
    if pipeline.simulator and hasattr(pipeline.simulator, 'theoretical_fisher_matrix'):
        print(f"\n{'='*60}")
        print("Comparison with Theoretical")
        print(f"{'='*60}")

        theta_fid_tensor = torch.tensor(config.analysis.theta_fid)
        F_theory = pipeline.simulator.theoretical_fisher_matrix(theta_fid_tensor)
        log_det_theory = torch.logdet(F_theory)
        constraints_theory = torch.sqrt(torch.diag(torch.linalg.inv(F_theory)))

        print(f"\n{'Method':<20} {'log|F|':>10} {'σ(A)':>10} {'σ(B)':>10}")
        print("-"*53)
        print(f"{'Theoretical':<20} {log_det_theory.cpu().item():>10.2f} {constraints_theory[0].cpu().item():>10.4f} {constraints_theory[1].cpu().item():>10.4f}")
        print(f"{'Pipeline':<20} {result.log_det_fisher.cpu().item():>10.2f} {result.constraints[0].cpu().item():>10.4f} {result.constraints[1].cpu().item():>10.4f}")

        ratio = (result.log_det_fisher.cpu() / log_det_theory.cpu()).item()
        print(f"\nEfficiency: {ratio:.1%} of theoretical maximum")

    # Save results if requested
    if args.save_results:
        output_dir = Path(config.experiment.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save expanded config with auto-selected values
        expanded_config = extract_expanded_config(config, pipeline)

        # Custom representer to use flow style for lists
        def represent_list(dumper, data):
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

        yaml.add_representer(list, represent_list)

        with open(output_dir / "config.yaml", 'w') as f:
            yaml.dump(expanded_config, f, default_flow_style=False, sort_keys=False)

        # Collect component representations
        component_metadata = {}
        if hasattr(pipeline, 'simulator'):
            component_metadata['simulator'] = repr(pipeline.simulator)
        if hasattr(pipeline, 'filtration'):
            component_metadata['filtration'] = repr(pipeline.filtration)
        if hasattr(pipeline, 'vectorization'):
            component_metadata['vectorization'] = repr(pipeline.vectorization)
        if hasattr(pipeline, 'compression'):
            component_metadata['compression'] = repr(pipeline.compression)
        if hasattr(pipeline, 'fisher_analyzer'):
            component_metadata['fisher_analyzer'] = repr(pipeline.fisher_analyzer)

        # Save results as JSON (move tensors to CPU first)
        results_dict = {
            'experiment': config.experiment.name,
            'fisher_matrix': result.fisher_matrix.cpu().tolist(),
            'log_det_fisher': result.log_det_fisher.cpu().item(),
            'constraints': result.constraints.cpu().tolist(),
            'is_gaussian': result.is_gaussian,
            'gaussianity_details': result.gaussianity_details,
            'analysis': {
                'theta_fid': config.analysis.theta_fid.tolist(),
                'delta_theta': config.analysis.delta_theta.tolist(),
                'n_s': config.analysis.n_s,
                'n_d': config.analysis.n_d
            },
            'metadata': {
                'components': component_metadata
            }
        }

        class NumpyEncoder(json.JSONEncoder):
            """Handle numpy types that aren't JSON serializable."""
            def default(self, obj):
                import numpy as np
                if isinstance(obj, (np.bool_,)):
                    return bool(obj)
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results_dict, f, indent=2, cls=NumpyEncoder)

        # Save Fisher matrix as numpy (move to CPU first)
        np.save(output_dir / 'fisher_matrix.npy', result.fisher_matrix.detach().cpu().numpy())

        # Save pipeline checkpoint (entire pipeline object to preserve hyperparameters)
        # This saves everything including auto-selected values (e.g., TopK k values)
        #
        # Convert CachedCompressionPipeline to BasePipeline for inference portability.
        # CachedCompressionPipeline requires cache config, but BasePipeline can run
        # with fresh data (new seeds) using any simulator.
        if isinstance(pipeline, CachedCompressionPipeline):
            # Load metadata from cache file to get simulator/filtration config
            cache_path = config.analysis.cache.load_path
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            metadata = cached_data.get('metadata', {})

            # Create simulator from cached metadata
            sim_meta = metadata.get('simulator', {})
            sim_type = sim_meta['type'].lower().replace('simulator', '')
            sim_params = sim_meta.get('params', {})
            simulator = create_simulator(SimulatorConfig(type=sim_type, params=sim_params))

            # Create filtration from cached metadata
            filt_meta = metadata.get('filtration', {})
            filt_type = filt_meta['type'].lower().replace('layer', '')
            filt_params = filt_meta.get('params', {})
            filtration = create_filtration(FiltrationConfig(type=filt_type, trainable=False, params=filt_params))

            inference_pipeline = BasePipeline(
                simulator=simulator,
                filtration=filtration,
                vectorization=pipeline.vectorization,
                compression=pipeline.compression,
                fisher_analyzer=pipeline.fisher_analyzer
            )
            print(f"  Converted CachedCompressionPipeline → BasePipeline for inference")
        else:
            inference_pipeline = pipeline

        # Save only model state_dict (not full pipeline) to avoid JAX pickling issues
        # Also save vectorization k values if they were auto-determined
        vectorization_state = {}
        if hasattr(inference_pipeline.vectorization, 'layers'):
            k_values = []
            for layer in inference_pipeline.vectorization.layers:
                if hasattr(layer, 'k'):
                    k_values.append(layer.k)
            if k_values:
                vectorization_state['k_values'] = k_values
        
        checkpoint = {
            'model_state_dict': inference_pipeline.state_dict(),
            'config_path': str(args.config),
            'vectorization_state': vectorization_state,
        }
        torch.save(checkpoint, output_dir / 'pipeline.pt')

        # Save training history if available
        if training_history is not None:
            # Convert NaN to null for JSON compatibility
            history_dict = {
                'train_losses': [x if not (isinstance(x, float) and np.isnan(x)) else None
                                 for x in training_history['train_losses']],
                'val_losses': [x if not (isinstance(x, float) and np.isnan(x)) else None
                               for x in training_history['val_losses']],
                'best_val_loss': training_history['best_val_loss'],
                'best_epoch': training_history['best_epoch'],
            }
            with open(output_dir / 'training_history.json', 'w') as f:
                json.dump(history_dict, f)
            print(f"  - training_history.json ({len(history_dict['train_losses'])} epochs)")

        print(f"\nResults saved to: {output_dir}")
        print(f"  - config.yaml, results.json, fisher_matrix.npy, pipeline.pt")

    print(f"\n{'='*60}")
    print("Pipeline completed successfully!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()