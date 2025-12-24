def train_epoch(model, trainloader, criterion, optimizer, device, is_snn=False):
    """Train one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    spike_rates = []
    
    pbar = tqdm(trainloader, desc="Training")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        if is_snn:
            outputs, spike_rate = model(inputs)
            spike_rates.append(spike_rate)
        else:
            outputs = model(inputs)
            spike_rate = 0
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.3f}',
            'Acc': f'{100.*correct/total:.2f}%',
            'Spike': f'{spike_rate:.4f}' if is_snn else 'N/A'
        })
    
    avg_loss = running_loss / len(trainloader)
    avg_acc = 100. * correct / total
    avg_spike = np.mean(spike_rates) if spike_rates else 0
    
    return avg_loss, avg_acc, avg_spike

def evaluate(model, testloader, criterion, device, is_snn=False):
    """Evaluate model on test set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    spike_rates = []
    
    with torch.no_grad():
        pbar = tqdm(testloader, desc="Testing")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            if is_snn:
                outputs, spike_rate = model(inputs)
                spike_rates.append(spike_rate)
            else:
                outputs = model(inputs)
                spike_rate = 0
            
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'Acc': f'{100.*correct/total:.2f}%',
                'Spike': f'{spike_rate:.4f}' if is_snn else 'N/A'
            })
    
    avg_loss = running_loss / len(testloader)
    avg_acc = 100. * correct / total
    avg_spike = np.mean(spike_rates) if spike_rates else 0
    
    return avg_loss, avg_acc, avg_spike

def train_model(model, trainloader, testloader, epochs=10, 
                model_type='qnn', lr=0.01):
    """Complete training pipeline"""
    criterion = nn.CrossEntropyLoss()
    
    if model_type == 'snn':
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        is_snn = True
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        is_snn = False
    
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    spike_rates = []
    
    print(f"\nTraining {model_type.upper()} for {epochs} epochs")
    print("=" * 50)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_acc, train_spike = train_epoch(
            model, trainloader, criterion, optimizer, device, is_snn
        )
        
        # Evaluate
        test_loss, test_acc, test_spike = evaluate(
            model, testloader, criterion, device, is_snn
        )
        
        # Record metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        if model_type == 'snn':
            spike_rates.append(test_spike)
        
        # Update scheduler
        scheduler.step()
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        if model_type == 'snn':
            print(f"Spike Rate: {test_spike:.4f}")
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'spike_rates': spike_rates
    }

