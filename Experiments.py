# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

import json
import os
from datetime import datetime

class ExperimentConfig:
    """Configuration for SNN vs QNN experiments"""
    
    def __init__(self, T_values=[6, 8, 10, 16], 
                 spike_rates=[0.01, 0.05, 0.1, 0.2, 0.3],
                 weight_bits_values=[2, 4, 8],
                 act_sparsity_values=[0.5, 0.7, 0.9],
                 epochs=50):
        
        self.T_values = T_values
        self.spike_rates = spike_rates
        self.weight_bits_values = weight_bits_values
        self.act_sparsity_values = act_sparsity_values
        self.epochs = epochs
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = f"experiments/exp_{timestamp}"
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Save config
        self.save_config()
        
        print(f"Experiment directory: {self.exp_dir}")
    
    def save_config(self):
        """Save configuration to JSON"""
        config_dict = {
            'T_values': self.T_values,
            'spike_rates': self.spike_rates,
            'weight_bits_values': self.weight_bits_values,
            'act_sparsity_values': self.act_sparsity_values,
            'epochs': self.epochs,
            'exp_dir': self.exp_dir
        }
        
        with open(os.path.join(self.exp_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def get_qnn_bits(self, T):
        """Get QNN bits for given T according to paper"""
        return get_qnn_bits_from_T(T)



def run_T_experiments(config):
    """Run experiments with different T values"""
    
    results = {}
    
    for T in config.T_values:
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: SNN T={T}, QNN bits={get_qnn_bits_from_T(T)}")
        print(f"{'='*70}")
        
        # Create SNN model
        snn_model = SNN_ResNet18_Proper(T=T).to(device)
        snn_params = print_model_summary(snn_model, f"SNN ResNet-18 (T={T})")
        
        # Create QNN model with paper-compliant bits
        qnn_bits = get_qnn_bits_from_T(T)
        qnn_model = QNN_ResNet18(act_bits=qnn_bits).to(device)
        qnn_params = print_model_summary(qnn_model, f"QNN ResNet-18 ({qnn_bits}-bit)")
        
        # Load data (reset for each experiment)
        trainloader, testloader = get_cifar10_loaders(batch_size=128)
        
        # Train SNN
        print(f"\nTraining SNN (T={T})...")
        snn_results = train_model(snn_model, trainloader, testloader,
                                 epochs=config.epochs, model_type='snn', lr=0.001)
        
        # Train QNN
        print(f"\nTraining QNN ({qnn_bits}-bit)...")
        qnn_results = train_model(qnn_model, trainloader, testloader,
                                 epochs=config.epochs, model_type='qnn', lr=0.01)
        
        # Energy estimation
        subset_loader = torch.utils.data.DataLoader(
            testloader.dataset, batch_size=64, shuffle=False, num_workers=0
        )
        
        snn_energy = estimate_inference_energy(snn_model, subset_loader, 'snn', T=T)
        qnn_energy = estimate_inference_energy(qnn_model, subset_loader, 'qnn')
        
        # Apply paper's analysis
        if 'spike_rate' in snn_energy:
            s_r = snn_energy['spike_rate']
            k = 0.5 / 0.08  # E_MAC_int8 / E_ACC
            
            # Theorem 2: Convert SNN spike rate to QNN sparsity
            gamma = 1 - (2 * s_r * T) / (T + 1)
            
            # Paper's condition for SNN advantage
            condition = T * s_r <= k * (1 - gamma)
        
        # Store results
        results[T] = {
            'T': T,
            'qnn_bits': qnn_bits,
            'snn_params': snn_params,
            'qnn_params': qnn_params,
            'snn_results': snn_results,
            'qnn_results': qnn_results,
            'snn_energy': snn_energy,
            'qnn_energy': qnn_energy,
            'paper_analysis': {
                'spike_rate': s_r if 'spike_rate' in snn_energy else None,
                'gamma': gamma if 'spike_rate' in snn_energy else None,
                'condition': condition if 'spike_rate' in snn_energy else None,
                'T_times_sr': T * s_r if 'spike_rate' in snn_energy else None,
                'k_times_one_minus_gamma': k * (1 - gamma) if 'spike_rate' in snn_energy else None
            }
        }
        
        # Save models and intermediate results
        torch.save(snn_model.state_dict(), 
                  os.path.join(config.exp_dir, f'snn_T{T}.pth'))
        torch.save(qnn_model.state_dict(),
                  os.path.join(config.exp_dir, f'qnn_bits{qnn_bits}.pth'))
        
        # Save intermediate results
        torch.save(results[T], 
                  os.path.join(config.exp_dir, f'results_T{T}.pth'))
        
        print(f"\n✓ Experiment T={T} completed and saved")
    
    # Save all results
    torch.save(results, os.path.join(config.exp_dir, 'all_results.pth'))
    
    return results


# ============================================================================
#               SPIKE RATE SWEEP EXPERIMENTS
# ============================================================================

def run_spike_rate_experiments(config, fixed_T=10):
    """Experiment with different spike rates for fixed T"""
    
    spike_rate_results = {}
    
    # We need to modify SNN to control spike rate
    # One approach: adjust threshold during inference
    
    print(f"\n{'='*70}")
    print(f"SPIKE RATE SWEEP: T={fixed_T}")
    print(f"{'='*70}")
    
    for target_sr in config.spike_rates:
        print(f"\nTarget Spike Rate: {target_sr}")
        
        # Create SNN model
        snn_model = SNN_ResNet18_Proper(T=fixed_T).to(device)
        
        # Adjust threshold to achieve target spike rate
        # Note: This is a simplification - in practice, you'd train with different thresholds
        adjusted_threshold = 1/ max(target_sr, 0.01)  # Simple heuristic
        
        # Create new model with adjusted threshold
        snn_model_threshold = SNN_ResNet18_Proper(T=fixed_T, v_threshold=adjusted_threshold).to(device)
        snn_model_threshold.load_state_dict(snn_model.state_dict())
        
        # Get QNN bits according to paper
        qnn_bits = get_qnn_bits_from_T(fixed_T)
        qnn_model = QNN_ResNet18(act_bits=qnn_bits).to(device)
        
        # Load data
        trainloader, testloader = get_cifar10_loaders(batch_size=128)
        
        # Quick evaluation (no full training)
        snn_model_threshold.eval()
        qnn_model.eval()
        
        criterion = nn.CrossEntropyLoss()
        
        # Measure SNN spike rate
        snn_acc = 0
        snn_total = 0
        measured_spike_rates = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(testloader, desc=f"Testing SR={target_sr}"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, spike_rate = snn_model_threshold(inputs)
                measured_spike_rates.append(spike_rate)
                
                _, predicted = outputs.max(1)
                snn_total += targets.size(0)
                snn_acc += predicted.eq(targets).sum().item()
        
        avg_spike_rate = np.mean(measured_spike_rates)
        
        # Energy estimation
        subset_loader = torch.utils.data.DataLoader(
            testloader.dataset, batch_size=64, shuffle=False, num_workers=0
        )
        
        snn_energy = estimate_inference_energy(snn_model_threshold, subset_loader, 'snn', T=fixed_T)
        qnn_energy = estimate_inference_energy(qnn_model, subset_loader, 'qnn')
        
        # Apply paper's analysis
        s_r = avg_spike_rate
        k = 0.5 / 0.08
        gamma = 1 - (2 * s_r * fixed_T) / (fixed_T + 1)
        condition = fixed_T * s_r <= k * (1 - gamma)
        
        spike_rate_results[target_sr] = {
            'target_sr': target_sr,
            'measured_sr': avg_spike_rate,
            'threshold': adjusted_threshold,
            'snn_acc': 100. * snn_acc / snn_total,
            'snn_energy': snn_energy,
            'qnn_energy': qnn_energy,
            'paper_analysis': {
                'gamma': gamma,
                'condition': condition,
                'T_times_sr': fixed_T * s_r,
                'k_times_one_minus_gamma': k * (1 - gamma),
                'energy_ratio': snn_energy['energy_per_sample_j'] / qnn_energy['energy_per_sample_j']
            }
        }
        
        print(f"  Measured SR: {avg_spike_rate:.4f}, SNN Acc: {100.*snn_acc/snn_total:.2f}%")
        print(f"  Energy Ratio (SNN/QNN): {snn_energy['energy_per_sample_j']/qnn_energy['energy_per_sample_j']:.3f}")
    
    # Save spike rate results
    torch.save(spike_rate_results, 
               os.path.join(config.exp_dir, f'spike_rate_results_T{fixed_T}.pth'))
    
    return spike_rate_results


# ============================================================================
#                 MAIN EXECUTION BLOCK
# ============================================================================


"""Run complete set of experiments"""
    
print("="*80)
print("COMPREHENSIVE SNN vs QNN EXPERIMENTS")
print("Paper: 'Reconsidering the Energy Efficiency of SNNs'")
print("="*80)
    
    # Create configuration
config = ExperimentConfig(
    T_values=[6,8,10,16],  # Test different T values
    spike_rates=[0.01, 0.05, 0.1, 0.2, 0.3],  # Different spike rates
    weight_bits_values=[2, 4, 8],  # Test different QNN precisions
    epochs= 50  # Training epochs
)
    
    # 1. Run T experiments
print("\n" + "="*80)
print("EXPERIMENT 1: Varying T Values")
print("="*80)
    
T_results = run_T_experiments(config)
    
    # 2. Run spike rate experiments
print("\n" + "="*80)
print("EXPERIMENT 2: Spike Rate Sweep (Fixed T=10)")
print("="*80)
    
spike_rate_results = run_spike_rate_experiments(config, fixed_T=10)
    
    # 3. Generate visualizations
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)
    
plot_T_comparison(T_results, config)
plot_spike_rate_analysis(spike_rate_results, config, fixed_T=10)
    
    # 4. Generate summary report
generate_summary_report(T_results, spike_rate_results, config)
    
print("\n" + "="*80)
print("EXPERIMENTS COMPLETED SUCCESSFULLY!")
print(f"All results saved in: {config.exp_dir}")
print("="*80)

