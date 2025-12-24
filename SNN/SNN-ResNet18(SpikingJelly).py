# ============================================================================
# ALTERNATIVE: USING SPIKINGJELLY'S PRE-BUILT LAYERS
# ============================================================================

class SNN_ResNet18_SpikingJelly(nn.Module):
    """SNN ResNet-18 using spikingjelly's pre-built layers"""
    def __init__(self, T=4, tau=2.0, v_threshold=1.0, num_classes=10):
        super().__init__()
        self.T = T
        
        # Use spikingjelly's sequential layer
        self.features = nn.Sequential(
            # Initial layers
            layer.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(64),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, 
                          surrogate_function=surrogate.ATan()),
            
            # Layer 1
            layer.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(64),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold),
            layer.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(64),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold),
            
            # Layer 2 (with stride)
            layer.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            layer.BatchNorm2d(128),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold),
            layer.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(128),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold),
            
            # Layer 3
            layer.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            layer.BatchNorm2d(256),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold),
            layer.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(256),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold),
            
            # Layer 4
            layer.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            layer.BatchNorm2d(512),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold),
            layer.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(512),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
        
        # Spike tracking
        self.spike_counts = []
    
    def reset_spike_counts(self):
        self.spike_counts = []
    
    def count_spikes(self, x):
        # Count spikes in all LIF layers
        for module in self.features.modules():
            if isinstance(module, neuron.BaseNode):
                self.spike_counts.append(module.spike.sum().item())
    
    def forward(self, x):
        batch_size = x.shape[0]
        self.reset_spike_counts()
        
        # Reset all neurons
        functional.reset_net(self)
        
        # Process T timesteps
        outputs = []
        x_prob = torch.clamp(x, 0, 1)  # Normalize for Poisson
        
        for t in range(self.T):
            # Generate Poisson spikes
            spikes_in = (torch.rand_like(x_prob) < x_prob).float()
            
            # Forward through spiking layers
            out = self.features(spikes_in)
            
            # Count spikes
            self.count_spikes(out)
            
            # Classify
            out = self.classifier(out)
            outputs.append(out)
        
        # Average outputs
        outputs = torch.stack(outputs, dim=1)
        final_output = outputs.mean(dim=1)
        
        # Calculate spike rate
        total_spikes = sum(self.spike_counts)
        # Estimate total neurons: ~11M params / ~100 = 110k neurons
        total_neurons = 110_000 * batch_size * self.T
        spike_rate = total_spikes / total_neurons if total_neurons > 0 else 0
        
        return final_output, spike_rate
