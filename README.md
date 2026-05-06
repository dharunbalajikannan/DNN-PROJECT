# Transfer Learning for Plant Disease Detection

This project explores the application of Transfer Learning using a pretrained **ResNet18** model to detect diseases in plants. The dataset used is the widely recognized **PlantVillage** dataset, which contains images of healthy and diseased plant leaves across 38 multiple categories.

## Project Objectives
- Implement transfer learning using a pretrained ResNet18 backbone.
- Experiment with different optimizers: **Adam**, **SGD**, and **RMSprop**.
- Study the effect of freezing backbone layers and adding custom classifier heads with Dropout.
- Evaluate model performance using Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and Training/Validation curves.

## Methodology
1. **Backbone**: ResNet18 (pretrained on ImageNet).
2. **Transfer Learning Strategy**: 
   - All layers in the ResNet backbone were frozen.
   - The final fully connected layer was replaced with a `Dropout(0.3)` layer followed by a `Linear` layer mapping to the 38 classes.
3. **Regularization**: Weight decay (L2) of `1e-4` and Dropout of `0.3`.
4. **Data Splitting**: 80% Training, 10% Validation, 10% Testing.
5. **Preprocessing**: 
   - Random Resized Crop (224x224).
   - Random Horizontal Flip and Rotation (15°).
   - Normalization using ImageNet mean and standard deviation.

## Results Table
The performance of the model using different optimizers (evaluated over 2 training epochs):

| Optimizer | Test Accuracy | Convergence Speed |
|-----------|---------------|-------------------|
| Adam      | 89.30%        | High              |
| RMSprop   | 88.67%        | High              |
| SGD       | 84.50%        | Medium            |

### Detailed Metrics (Best Model - Adam)
The best performing configuration used the **Adam** optimizer. The detailed classification metrics on the test set are as follows:
- **Accuracy**: 90.56%
- **Macro Precision**: 91.28%
- **Macro Recall**: 88.49%
- **Macro F1-Score**: 89.55%

## Visualizations
The performance of each optimizer was recorded and compared. You can find the following visualizations in the `results/` folder:
- **`training_history.png`**: Accuracy and Loss convergence curves for the Adam optimizer.
- **`val_acc_comparison.png`** / **`val_loss_comparison.png`**: Comparative graphs showing how the validation accuracy and loss scaled over epochs for Adam, SGD, and RMSprop.
- **`confusion_matrix.png`**: Detailed heatmap view of class-wise prediction performance for the best model.
- **`sample_predictions.png`**: Randomly selected test images showcasing predicted vs actual plant disease classifications.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the full experiment comparison to generate the models:
   ```bash
   python experiments.py
   ```
3. Generate the confusion matrix and final precision/recall metrics (requires a trained model):
   ```bash
   python generate_visuals.py
   ```

## Repository Structure
- `data_loader.py`: Handles dataset loading, splitting, and ImageNet standardization.
- `model.py`: Defines the ResNet18 model architecture and classification head.
- `trainer.py`: Implementation of the training loop and evaluation logic.
- `visualize.py`: Helper script for generating plots and charts.
- `experiments.py`: Comparative study across different optimizers.
- `generate_visuals.py`: Final report script that computes classification reports and generates sample predictions.
- `results/`: Directory containing saved plots, models, and metric JSON files.
