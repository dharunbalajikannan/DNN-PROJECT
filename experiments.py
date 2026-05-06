import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_data_loaders
from model import get_resnet_model
from trainer import train_model, evaluate_model
from visualize import plot_comparative_history, plot_history
import os
import json

# Configuration
DATA_DIR = "C:/Users/ShashanK/OneDrive/Documents/OP/Machine Learning/FineTuning/PlantVillage/PlantVillage"
BATCH_SIZE = 64
EPOCHS = 2
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

def run_experiments():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Data
    train_loader, val_loader, test_loader, num_classes, class_names = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)
    dataloaders = {'train': train_loader, 'val': val_loader}

    optimizers_to_test = ['Adam', 'SGD', 'RMSprop']
    all_histories = {}
    test_results = {}

    for opt_name in optimizers_to_test:
        print(f"\n{'='*20}\nRunning experiment with: {opt_name}\n{'='*20}")
        
        # Initialize model
        model = get_resnet_model(num_classes=num_classes, pretrained=True, freeze_layers=True).to(device)
        
        # Setup Optimizer
        if opt_name == 'Adam':
            optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        elif opt_name == 'SGD':
            optimizer = optim.SGD(model.fc.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
        elif opt_name == 'RMSprop':
            optimizer = optim.RMSprop(model.fc.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        criterion = nn.CrossEntropyLoss()

        # Train
        model, history = train_model(model, dataloaders, criterion, optimizer, device, num_epochs=EPOCHS)
        all_histories[opt_name] = history

        # Evaluate on Test Set
        test_acc, y_true, y_pred = evaluate_model(model, test_loader, device)
        test_results[opt_name] = test_acc
        
        # Save model
        save_dir = "results"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_{opt_name}.pth"))

    # Save results summary
    with open(os.path.join(save_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=4)

    # Plot comparisons
    plot_comparative_history(all_histories, metric='val_acc', save_path=os.path.join(save_dir, "val_acc_comparison.png"))
    plot_comparative_history(all_histories, metric='val_loss', save_path=os.path.join(save_dir, "val_loss_comparison.png"))

    print("\nExperiments completed successfully!")
    for opt, acc in test_results.items():
        print(f"{opt} Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    run_experiments()
