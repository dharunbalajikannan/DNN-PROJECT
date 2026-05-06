# Model Training Summary - Plant Disease Detection

## Model Hyperparameters
- **Backbone**: ResNet18 (Freezed)
- **Classifier**: Dropout(0.3) -> Linear(FT, Classes)
- **Batch Size**: 64
- **Initial Learning Rate**: 0.001
- **Weight Decay**: 0.0001
- **Epochs per Optimizer**: 2

## Comparative Results (Optimizer Study)
| Optimizer | Test Accuracy | Convergence Speed |
|-----------|---------------|-------------------|
| Adam      | 89.30%        | High              |
| SGD       | 84.50%        | Medium            |
| RMSprop   | 88.67%        | High              |

## Best Performing Configuration
- **Optimizer**: Adam
- **Peak Validation Accuracy**: ~89.3%
- **Test Set Accuracy**: 89.30%

*Note: Visualizations available in the results folder.*
