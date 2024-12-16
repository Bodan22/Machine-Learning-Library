# Machine Learning Framework

A simple Java-based machine learning library for binary classification with a JavaFX GUI interface. This framework implements various classification algorithms and evaluation metrics, designed according to object-oriented principles.

## Features

### Classification Algorithms
- K-Nearest Neighbors (KNN)
  - Configurable k value
  - Support for different distance metrics (Euclidean, Manhattan, Custom Diabetes metric)
- Logistic Regression
  - Configurable learning rate and epochs
  - Feature normalization
- Decision Tree
  - Information gain-based splitting
  - Automatic handling of numerical features

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### GUI Features
- File selection for dataset input
- Model selection and hyperparameter configuration
- Adjustable train-test split ratio
- Real-time training and evaluation
- Results visualization including confusion matrix

  
## Requirements

- Java 21 or higher
- JavaFX 23.0.1 or higher

## Setup and Running

1. Clone the repository
2. Ensure all required libraries are in your classpath
3. Set up JavaFX:
   ```
   --module-path /path/to/javafx-sdk/lib --add-modules javafx.controls,javafx.fxml,javafx.graphics
   ```
4. Run the GUI.java class

## Usage

1. Launch the application
2. Select your input dataset (CSV format)
3. Choose a classification algorithm:
   - KNN: Set k value and distance metric
   - Logistic Regression: Configure learning rate and epochs
   - Decision Tree: No additional parameters required
4. Set the train-test split ratio using the slider
5. Click "Train and Evaluate Model"
6. View results including:
   - Accuracy, Precision, Recall, and F1 Score
   - Confusion Matrix visualization

## Implementation Details

### Models
All models implement the `Model<F, L>` interface with methods:
- `train(List<Instance<F, L>> instances)`
- `test(List<Instance<F, L>> instances)`

### Evaluation
Metrics are implemented as separate utility classes with static methods for:
- Calculating metric values
- Formatting results
- Handling edge cases



## Contributing
Feel free to contribute by:
- Implementing additional classification algorithms
- Adding new evaluation metrics
- Improving the GUI
- Optimizing existing implementations
