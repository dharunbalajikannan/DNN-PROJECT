import torch
from data_loader import get_data_loaders
from model import get_resnet_model
from trainer import evaluate_model
from visualize import plot_confusion_matrix, plot_history
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def show_sample_predictions(model, test_loader, class_names, device, num_images=5):
    """
    Displays a few images with their predicted and actual labels.
    """
    model.eval()
    images_so_far = 0
    plt.figure(figsize=(15, 10))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 5 + 1, 5, images_so_far)
                ax.axis('off')
                ax.set_title(f'P: {class_names[preds[j]]}\nA: {class_names[labels[j]]}', fontsize=8)
                
                # Unnormalize image for visualization
                img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                plt.imshow(img)

                if images_so_far == num_images:
                    plt.tight_layout()
                    plt.savefig("results/sample_predictions.png")
                    plt.show()
                    return

def main():
    # To run the full experiment (SGD, Adam, RMSprop)
    # from experiments import run_experiments
    # run_experiments()

    # For now, let's run a single high-quality training with Adam to get the best model
    # and show the final visualizations as requested.
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = "C:/Users/ShashanK/OneDrive/Documents/OP/Machine Learning/FineTuning/PlantVillage/PlantVillage"
    results_dir = "results"
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print("Initializing Data Loaders...")
    train_loader, val_loader, test_loader, num_classes, class_names = get_data_loaders(DATA_DIR, batch_size=64)
    dataloaders = {'train': train_loader, 'val': val_loader}

    print(f"Loading Pretrained ResNet18 for {num_classes} classes...")
    model = get_resnet_model(num_classes, pretrained=True, freeze_layers=True, dropout=0.3).to(device)

    import torch.optim as optim
    import torch.nn as nn
    from trainer import train_model

    optimizer = optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("Starting Training (Adam Optimizer)...")
    model, history = train_model(model, dataloaders, criterion, optimizer, device, num_epochs=3)

    print("Evaluating on Test Set...")
    test_acc, y_true, y_pred = evaluate_model(model, test_loader, device)

    print("Generating Visualizations...")
    plot_history(history, "Adam", save_path=os.path.join(results_dir, "training_history.png"))
    plot_confusion_matrix(y_true, y_pred, class_names, save_path=os.path.join(results_dir, "confusion_matrix.png"))
    show_sample_predictions(model, test_loader, class_names, device)

    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"Results saved in '{results_dir}' directory.")

if __name__ == "__main__":
    main()
