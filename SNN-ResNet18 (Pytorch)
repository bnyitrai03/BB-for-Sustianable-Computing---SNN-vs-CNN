import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional, surrogate, layer



# ============================================================================
# 1. PROPER LIF NEURON WITH RESET
# ============================================================================

class LIFNeuronProper(neuron.BaseNode):
    """Proper LIF neuron with reset-to-zero"""
    def __init__(self, tau=2.0, v_threshold=1.0, surrogate_function=surrogate.ATan()):
        super().__init__(v_threshold=v_threshold, v_reset=0.0, 
                        surrogate_function=surrogate_function, detach_reset=True)
        self.tau = tau
    
    def neuronal_charge(self, x: torch.Tensor):
        # Leaky integrate: v = v * decay + x
        decay = torch.exp(torch.tensor(-1.0 / self.tau))
        self.v = self.v * decay + x

# ============================================================================
# 2. PROPER SPIKING RESNET BLOCK
# ============================================================================

class SpikingBasicBlock(nn.Module):
    """Proper spiking ResNet basic block with temporal processing"""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, tau=2.0, v_threshold=1.0):
        super().__init__()
        
        # First conv + BN + LIF
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.lif1 = LIFNeuronProper(tau=tau, v_threshold=v_threshold)
        
        # Second conv + BN + LIF
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.lif2 = LIFNeuronProper(tau=tau, v_threshold=v_threshold)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
            # Note: Shortcut doesn't have LIF - it's a direct connection
    
    def forward(self, x):
        """
        x: input spikes at current timestep
        Returns: output spikes at current timestep
        """
        # Save identity for residual
        identity = x
        
        # First conv + BN + LIF
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lif1(out)  # This outputs spikes
        
        # Second conv + BN + LIF
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.lif2(out)  # This outputs spikes
        
        # Shortcut (no LIF - we add membrane potentials, not spikes)
        if len(self.shortcut) > 0:
            identity = self.shortcut(identity)
        
        # Residual connection: Add identity to membrane potential before spiking
        # We add to membrane potential, not spikes
        self.lif2.v = self.lif2.v + identity
        
        # Generate spikes from updated membrane potential
        out = self.lif2.neuronal_fire()
        
        return out

# ============================================================================
# 3. PROPER SNN RESNET-18 WITH TEMPORAL PROCESSING
# ============================================================================

class SNN_ResNet18_Proper(nn.Module):
    """Proper SNN ResNet-18 with exact T timestep processing"""
    def __init__(self, T=4, tau=2.0, v_threshold=1.0, num_classes=10):
        super().__init__()
        self.T = T  # Number of timesteps
        self.tau = tau
        self.v_threshold = v_threshold
        
        self.in_planes = 64
        
        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.lif1 = LIFNeuronProper(tau=tau, v_threshold=v_threshold)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # Readout layers (non-spiking)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * SpikingBasicBlock.expansion, num_classes)
        
        # Spike tracking
        self.spike_counts = []
    
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(SpikingBasicBlock(self.in_planes, planes, stride, 
                                          self.tau, self.v_threshold))
            self.in_planes = planes * SpikingBasicBlock.expansion
        return nn.Sequential(*layers)
    
    def reset_spike_counts(self):
        """Reset spike counters"""
        self.spike_counts = []
    
    def count_spikes(self, x):
        """Count spikes in tensor"""
        self.spike_counts.append(x.sum().item())
    
    def forward(self, x):
        """
        x: input tensor of shape (batch, 3, 32, 32)
        Process over T timesteps with rate coding
        """
        batch_size = x.shape[0]
        
        # Reset spike counts
        self.reset_spike_counts()
        
        # Initialize membrane potentials
        functional.reset_net(self)
        
        # Rate coding: Convert input to Poisson spike trains
        # Scale input to firing probability (0-1)
        x_prob = torch.clamp(x, 0, 1)  # Normalize to [0, 1]
        
        # Process T timesteps
        outputs = []
        
        for t in range(self.T):
            # Generate Poisson spikes for this timestep
            spike_mask = torch.rand_like(x_prob) < x_prob
            spikes_in = spike_mask.float()
            
            # Initial conv + BN
            out = self.conv1(spikes_in)
            out = self.bn1(out)
            out = self.lif1(out)  # Output spikes
            self.count_spikes(out)
            
            # ResNet layers
            out = self.layer1(out)
            self.count_spikes(out)
            out = self.layer2(out)
            self.count_spikes(out)
            out = self.layer3(out)
            self.count_spikes(out)
            out = self.layer4(out)
            self.count_spikes(out)
            
            # Average pool and classify (non-spiking)
            out_pool = self.avgpool(out)
            out_pool = torch.flatten(out_pool, 1)
            out_class = self.fc(out_pool)
            
            outputs.append(out_class)
        
        # Average outputs over timesteps
        outputs = torch.stack(outputs, dim=1)  # (batch, T, num_classes)
        final_output = outputs.mean(dim=1)
        
        # Calculate spike rate
        total_spikes = sum(self.spike_counts)
        total_neurons = len(self.spike_counts) * batch_size * 512 * 1 * 1  # Last layer
        spike_rate = total_spikes / (total_neurons * self.T) if total_neurons > 0 else 0
        
        return final_output, spike_rate
