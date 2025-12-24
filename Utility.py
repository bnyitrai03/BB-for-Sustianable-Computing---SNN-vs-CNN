def get_qnn_bits_from_T(T):
    """According to paper Theorem 1: QNN bits = ceil(log2(T+1))"""
    import math
    return math.ceil(math.log2(T + 1))

# Test the function
for T in [1, 2, 4, 8, 16, 32]:
    bits = get_qnn_bits_from_T(T)
    print(f"SNN T={T} → Equivalent QNN bits={bits}")


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

import matplotlib.pyplot as plt
import seaborn as sns

def plot_T_comparison(results, config):
    """Plot comparison across different T values"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SNN vs QNN Comparison Across Different T Values', fontsize=16, fontweight='bold')
    
    # Prepare data
    T_values = sorted(results.keys())
    snn_accs = [results[T]['snn_results']['test_accs'][-1] for T in T_values]
    qnn_accs = [results[T]['qnn_results']['test_accs'][-1] for T in T_values]
    snn_energies = [results[T]['snn_energy']['energy_per_sample_j']*1e6 for T in T_values]
    qnn_energies = [results[T]['qnn_energy']['energy_per_sample_j']*1e6 for T in T_values]
    qnn_bits = [results[T]['qnn_bits'] for T in T_values]
    
    # 1. Accuracy vs T
    ax1 = axes[0, 0]
    ax1.plot(T_values, snn_accs, 'ro-', linewidth=2, markersize=8, label='SNN')
    ax1.plot(T_values, qnn_accs, 'bs-', linewidth=2, markersize=8, label='QNN')
    ax1.set_xlabel('SNN Timesteps (T)', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy vs T', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add QNN bits annotation
    for i, (x, y, bits) in enumerate(zip(T_values, qnn_accs, qnn_bits)):
        ax1.annotate(f'{bits}b', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    # 2. Energy vs T
    ax2 = axes[0, 1]
    ax2.plot(T_values, snn_energies, 'ro-', linewidth=2, markersize=8, label='SNN')
    ax2.plot(T_values, qnn_energies, 'bs-', linewidth=2, markersize=8, label='QNN')
    ax2.set_xlabel('SNN Timesteps (T)', fontsize=12)
    ax2.set_ylabel('Energy per sample (μJ)', fontsize=12)
    ax2.set_title('Energy Consumption vs T', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Energy Ratio (SNN/QNN) vs T
    ax3 = axes[0, 2]
    energy_ratios = [snn_e/qnn_e for snn_e, qnn_e in zip(snn_energies, qnn_energies)]
    ax3.plot(T_values, energy_ratios, 'g^-', linewidth=2, markersize=8)
    ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Break-even')
    ax3.set_xlabel('SNN Timesteps (T)', fontsize=12)
    ax3.set_ylabel('Energy Ratio (SNN/QNN)', fontsize=12)
    ax3.set_title('Energy Efficiency Ratio', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Fill where SNN is better
    ax3.fill_between(T_values, 0, 1, where=np.array(energy_ratios) < 1, 
                     alpha=0.3, color='green', label='SNN Advantage')
    
    # 4. Paper's Condition Plot
    ax4 = axes[1, 0]
    
    # Plot T*s_r vs k*(1-γ) for each T
    T_sr_values = []
    k_one_minus_gamma_values = []
    
    for T in T_values:
        if results[T]['paper_analysis']['spike_rate']:
            s_r = results[T]['paper_analysis']['spike_rate']
            gamma = results[T]['paper_analysis']['gamma']
            k = 0.5 / 0.08
            
            T_sr_values.append(T * s_r)
            k_one_minus_gamma_values.append(k * (1 - gamma))
    
    if T_sr_values:
        x_vals = list(range(len(T_values)))
        width = 0.35
        
        ax4.bar([x - width/2 for x in x_vals], T_sr_values, width, 
                label='T × s_r', color='orange', alpha=0.8)
        ax4.bar([x + width/2 for x in x_vals], k_one_minus_gamma_values, width,
                label='k × (1-γ)', color='blue', alpha=0.8)
        
        ax4.set_xlabel('T Value', fontsize=12)
        ax4.set_ylabel('Value', fontsize=12)
        ax4.set_title("Paper's Condition: T×s_r vs k×(1-γ)", fontsize=14)
        ax4.set_xticks(x_vals)
        ax4.set_xticklabels(T_values)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Spike Rate Evolution for each T
    ax5 = axes[1, 1]
    for T in T_values:
        if results[T]['snn_results']['spike_rates']:
            spike_rates = results[T]['snn_results']['spike_rates']
            epochs = range(1, len(spike_rates) + 1)
            ax5.plot(epochs, spike_rates, '-', linewidth=2, label=f'T={T}')
    
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Spike Rate', fontsize=12)
    ax5.set_title('Spike Rate Evolution', fontsize=14)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Training Loss Comparison
    ax6 = axes[1, 2]
    for T in T_values[:3]:  # Show first 3 for clarity
        snn_losses = results[T]['snn_results']['test_losses']
        qnn_losses = results[T]['qnn_results']['test_losses']
        epochs = range(1, len(snn_losses) + 1)
        
        ax6.plot(epochs, snn_losses, '-', linewidth=2, label=f'SNN T={T}')
        ax6.plot(epochs, qnn_losses, '--', linewidth=2, label=f'QNN {results[T]["qnn_bits"]}b')
    
    ax6.set_xlabel('Epoch', fontsize=12)
    ax6.set_ylabel('Test Loss', fontsize=12)
    ax6.set_title('Training Loss Comparison', fontsize=14)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.exp_dir, 'T_comparison_plots.png'), dpi=150, bbox_inches='tight')
    plt.show()

def plot_spike_rate_analysis(spike_rate_results, config, fixed_T=4):
    """Plot spike rate sweep analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Spike Rate Analysis (T={fixed_T})', fontsize=16, fontweight='bold')
    
    # Prepare data
    target_srs = sorted(spike_rate_results.keys())
    measured_srs = [spike_rate_results[sr]['measured_sr'] for sr in target_srs]
    energy_ratios = [spike_rate_results[sr]['paper_analysis']['energy_ratio'] for sr in target_srs]
    T_times_sr = [spike_rate_results[sr]['paper_analysis']['T_times_sr'] for sr in target_srs]
    k_one_minus_gamma = [spike_rate_results[sr]['paper_analysis']['k_times_one_minus_gamma'] for sr in target_srs]
    
    # 1. Target vs Measured Spike Rate
    ax1 = axes[0, 0]
    ax1.plot(target_srs, measured_srs, 'go-', linewidth=2, markersize=8)
    ax1.plot([0, max(target_srs)], [0, max(target_srs)], 'r--', alpha=0.5, label='Ideal')
    ax1.set_xlabel('Target Spike Rate', fontsize=12)
    ax1.set_ylabel('Measured Spike Rate', fontsize=12)
    ax1.set_title('Target vs Measured Spike Rates', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Energy Ratio vs Spike Rate
    ax2 = axes[0, 1]
    ax2.plot(measured_srs, energy_ratios, 'bo-', linewidth=2, markersize=8)
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Break-even')
    ax2.set_xlabel('Measured Spike Rate', fontsize=12)
    ax2.set_ylabel('Energy Ratio (SNN/QNN)', fontsize=12)
    ax2.set_title('Energy Efficiency vs Spike Rate', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Fill SNN advantage region
    measured_array = np.array(measured_srs)
    ratio_array = np.array(energy_ratios)
    ax2.fill_between(measured_array, 0, 1, where=ratio_array < 1, 
                     alpha=0.3, color='green', label='SNN Advantage')
    
    # 3. Paper's Condition
    ax3 = axes[1, 0]
    ax3.plot(measured_srs, T_times_sr, 'o-', linewidth=2, markersize=8, label='T × s_r')
    ax3.plot(measured_srs, k_one_minus_gamma, 's-', linewidth=2, markersize=8, label='k × (1-γ)')
    ax3.set_xlabel('Spike Rate (s_r)', fontsize=12)
    ax3.set_ylabel('Value', fontsize=12)
    ax3.set_title("Paper's Condition Analysis", fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Highlight where condition holds
    condition_holds = [T*sr <= k_g for T, sr, k_g in zip([fixed_T]*len(measured_srs), measured_srs, k_one_minus_gamma)]
    for i, (x, y, holds) in enumerate(zip(measured_srs, T_times_sr, condition_holds)):
        color = 'green' if holds else 'red'
        ax3.scatter(x, y, color=color, s=100, zorder=5, edgecolors='black')
    
    # 4. Energy-Accuracy Trade-off
    ax4 = axes[1, 1]
    snn_accs = [spike_rate_results[sr]['snn_acc'] for sr in target_srs]
    
    scatter = ax4.scatter(measured_srs, snn_accs, c=energy_ratios, 
                         cmap='RdYlGn', s=100, alpha=0.8, edgecolors='black')
    
    ax4.set_xlabel('Spike Rate', fontsize=12)
    ax4.set_ylabel('SNN Accuracy (%)', fontsize=12)
    ax4.set_title('Energy-Accuracy Trade-off', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Energy Ratio (SNN/QNN)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.exp_dir, f'spike_rate_analysis_T{fixed_T}.png'), 
                dpi=150, bbox_inches='tight')
    plt.show()


def generate_summary_report(T_results, spike_rate_results, config):
    """Generate text summary report"""
    
    report_path = os.path.join(config.exp_dir, 'experiment_summary.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("EXPERIMENT SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("EXPERIMENT CONFIGURATION:\n")
        f.write("-"*40 + "\n")
        f.write(f"T values tested: {config.T_values}\n")
        f.write(f"Spike rates tested: {config.spike_rates}\n")
        f.write(f"Training epochs: {config.epochs}\n\n")
        
        f.write("MODEL ARCHITECTURE:\n")
        f.write("-"*40 + "\n")
        f.write("Both models: ResNet-18\n")
        f.write("Input: CIFAR-10 (32x32 RGB)\n")
        f.write("Output: 10 classes\n\n")
        
        f.write("RESULTS BY T VALUE:\n")
        f.write("-"*40 + "\n")
        
        for T in sorted(T_results.keys()):
            res = T_results[T]
            f.write(f"\nT = {T} (QNN bits = {res['qnn_bits']}):\n")
            f.write(f"  SNN Accuracy: {res['snn_results']['test_accs'][-1]:.2f}%\n")
            f.write(f"  QNN Accuracy: {res['qnn_results']['test_accs'][-1]:.2f}%\n")
            f.write(f"  SNN Energy: {res['snn_energy']['energy_per_sample_j']*1e6:.2f} uJ\n")
            f.write(f"  QNN Energy: {res['qnn_energy']['energy_per_sample_j']*1e6:.2f} uJ\n")
            f.write(f"  Energy Ratio (SNN/QNN): {res['snn_energy']['energy_per_sample_j']/res['qnn_energy']['energy_per_sample_j']:.3f}\n")
            
            if res['paper_analysis']['condition'] is not None:
                condition = "✓ Holds" if res['paper_analysis']['condition'] else "✗ Fails"
                f.write(f"  Paper's condition: {condition}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("="*80 + "\n")
        
        # Find best SNN configuration
        best_snn_T = None
        best_snn_energy_ratio = float('inf')
        
        for T, res in T_results.items():
            energy_ratio = res['snn_energy']['energy_per_sample_j'] / res['qnn_energy']['energy_per_sample_j']
            if energy_ratio < best_snn_energy_ratio:
                best_snn_energy_ratio = energy_ratio
                best_snn_T = T
        
        if best_snn_T is not None:
            f.write(f"\n1. Most energy-efficient SNN: T={best_snn_T}\n")
            f.write(f"   Energy savings: {(1-best_snn_energy_ratio)*100:.1f}% vs equivalent QNN\n")
        
        # Check paper's condition
        f.write("\n2. Paper's Condition Analysis:\n")
        for T, res in T_results.items():
            if res['paper_analysis']['condition'] is not None:
                condition = "Holds" if res['paper_analysis']['condition'] else "Fails"
                f.write(f"   T={T}: Condition {condition} ")
                if res['paper_analysis']['condition']:
                    f.write("(SNN should be efficient)\n")
                else:
                    f.write("(QNN should be efficient)\n")
        
        f.write("\n3. Spike Rate Observations:\n")
        for T, res in T_results.items():
            if res['paper_analysis']['spike_rate']:
                s_r = res['paper_analysis']['spike_rate']
                f.write(f"   T={T}: s_r={s_r:.4f}, T×s_r={T*s_r:.4f}\n")
        
        f.write("\n4. Practical Recommendations:\n")
        f.write("   - For mobile edge devices, consider T ≤ 4 for SNNs\n")
        f.write("   - Maintain spike rate s_r < 10% for energy efficiency\n")
        f.write("   - Use QNN when high precision needed (8-bit or more)\n")
        f.write("   - Use SNN for ultra-low-power applications\n")
    
    print(f"Summary report saved: {report_path}")


# ============================================================================
#                          MODEL ANALYSIS FUNCTIONS
# ============================================================================

def count_model_params(model):
    """Count total parameters in model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_params_m': total_params / 1e6,
        'model_size_mb': (total_params * 4) / (1024**2)  # FP32 size
    }

def print_model_summary(model, model_name):
    """Print detailed model summary"""
    print(f"\n{'='*60}")
    print(f"MODEL SUMMARY: {model_name}")
    print(f"{'='*60}")
    
    params_info = count_model_params(model)
    print(f"Total parameters: {params_info['total_params']:,}")
    print(f"Trainable parameters: {params_info['trainable_params']:,}")
    print(f"Model size (FP32): {params_info['model_size_mb']:.2f} MB")
    
    # Layer-wise breakdown
    print(f"\n{'Layer':<25} {'Parameters':<15} {'Shape':<20}")
    print(f"{'-'*60}")
    
    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            print(f"{name:<25} {param.numel():<15,} {str(tuple(param.shape)):<20}")
    
    return params_info





