import torch
from data_loader import get_data_loaders
from model import get_resnet_model
from trainer import evaluate_model
from visualize import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import json

def show_sample_predictions(model, test_loader, class_names, device, num_images=10):
    model.eval()
    images_so_far = 0
    plt.figure(figsize=(20, 12))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(2, 5, images_so_far)
                ax.axis('off')
                ax.set_title(f'P: {class_names[preds[j]]}\nA: {class_names[labels[j]]}', fontsize=9)
                
                img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                plt.imshow(img)

                if images_so_far == num_images:
                    plt.tight_layout()
                    plt.savefig("results/sample_predictions.png")
                    print("Sample predictions saved.")
                    return

def generate_final_visuals():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = "C:/Users/ShashanK/OneDrive/Documents/OP/Machine Learning/FineTuning/PlantVillage/PlantVillage"
    results_dir = "results"
    
    # Load data
    _, _, test_loader, num_classes, class_names = get_data_loaders(DATA_DIR, batch_size=64)
    
    # Load best model (Adam)
    model = get_resnet_model(num_classes, pretrained=True, freeze_layers=True).to(device)
    model.load_state_dict(torch.load(os.path.join(results_dir, "model_Adam.pth")))
    
    print("Generating Confusion Matrix and Detailed Metrics...")
    _, y_true, y_pred = evaluate_model(model, test_loader, device)
    
    # Calculate detailed metrics
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Save metrics to JSON
    with open(os.path.join(results_dir, "detailed_metrics.json"), "w") as f:
        json.dump(report, f, indent=4)
    
    print("\nDetailed Metrics:")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
    print(f"Macro Precision: {report['macro avg']['precision']:.4f}")
    print(f"Macro Recall: {report['macro avg']['recall']:.4f}")

    plot_confusion_matrix(y_true, y_pred, class_names, save_path=os.path.join(results_dir, "confusion_matrix.png"))
    
    print("Generating Sample Predictions...")
    show_sample_predictions(model, test_loader, class_names, device)

if __name__ == "__main__":
    generate_final_visuals()
