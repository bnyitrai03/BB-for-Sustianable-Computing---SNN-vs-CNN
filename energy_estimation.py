def estimate_inference_energy(model, testloader, model_type='qnn', T=4):
    """Estimate inference energy using paper's framework"""
    model.eval()
    
    # Energy parameters (from paper, adjusted for mobile)
    E_MAC_int8 = 0.5    # pJ per INT8 MAC (mobile NPU)
    E_ACC = 0.08        # pJ per addition
    E_mem_dense = 0.02  # pJ per bit (dense)
    E_mem_sparse = 0.10 # pJ per bit (sparse, with overhead)
    E_weight = 0.15     # pJ per weight access
    
    if model_type == 'qnn':
        # QNN: 8-bit weights and activations
        bits = 8
        total_macs = 0
        total_energy = 0
        
        with torch.no_grad():
            for inputs, _ in tqdm(testloader, desc="QNN Energy"):
                inputs = inputs.to(device)
                
                # Estimate MACs (simplified: count parameters * spatial dimensions)
                # ResNet-18 on CIFAR-10: ~11M params, each used multiple times
                batch_size = inputs.shape[0]
                # Rough estimate: each param used ~H*W times
                # For 32x32 input after pooling: ~8x8 = 64 spatial average
                macs_per_sample = 11_000_000 * 64  # ~704M MACs per sample
                batch_macs = macs_per_sample * batch_size
                total_macs += batch_macs
                
                # Compute energy
                compute_energy = batch_macs * E_MAC_int8
                
                # Memory energy (weights)
                weight_energy = 11_000_000 * bits * E_weight * batch_size
                
                # Activation memory (simplified)
                activation_energy = batch_macs * bits * E_mem_dense
                
                batch_energy = compute_energy + weight_energy + activation_energy
                total_energy += batch_energy
        
        energy_per_sample = total_energy / len(testloader.dataset)
        
        return {
            'total_energy_j': total_energy / 1e12,
            'energy_per_sample_j': energy_per_sample / 1e12,
            'total_macs': total_macs,
            'model_type': 'QNN',
            'bits': bits
        }
    
    else:  # SNN
        bits = 1
        total_spikes = 0
        total_energy = 0
        spike_rates = []
        
        with torch.no_grad():
            for inputs, _ in tqdm(testloader, desc="SNN Energy"):
                inputs = inputs.to(device)
                batch_size = inputs.shape[0]
                
                # Forward pass to get spike rate
                outputs, spike_rate = model(inputs)
                spike_rates.append(spike_rate)
                
                # Estimate total spikes
                # ResNet-18 has ~11M parameters, but not all are neurons
                # Estimate ~1M neurons in the network
                num_neurons = 1_000_000
                batch_spikes = spike_rate * num_neurons * T * batch_size
                total_spikes += batch_spikes
                
                # SNN energy calculation
                # Compute: accumulations for each spike
                compute_energy = batch_spikes * E_ACC
                
                # Memory: weights accessed T times (worst case, dense)
                weight_energy = 11_000_000 * bits * E_weight * T * batch_size
                
                # Spike memory: each spike stored
                spike_memory_energy = batch_spikes * bits * E_mem_sparse
                
                batch_energy = compute_energy + weight_energy + spike_memory_energy
                total_energy += batch_energy
        
        energy_per_sample = total_energy / len(testloader.dataset)
        avg_spike_rate = np.mean(spike_rates)
        
        return {
            'total_energy_j': total_energy / 1e12,
            'energy_per_sample_j': energy_per_sample / 1e12,
            'total_spikes': total_spikes,
            'spike_rate': avg_spike_rate,
            'model_type': 'SNN',
            'T': T,
            'bits': bits
        }

