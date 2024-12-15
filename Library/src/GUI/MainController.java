package GUI;

import DataProcessing.CSV.CSVConvert;
import DataProcessing.domain.Instance;
import DataProcessing.domain.Model;
import Models.DecisionTree;
import Models.KNNClassifier;
import Models.LogisticRegression;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.stage.FileChooser;
import utils.DiabetesDistanceMetric;
import utils.DistanceMetric;
import utils.EuclideanDistance;
import utils.ManhattanDistance;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class MainController {
    @FXML private TextField filePathField;
    @FXML private ComboBox<String> classifierComboBox;
    @FXML private VBox knnConfig;
    @FXML private VBox logisticConfig;
    @FXML private VBox treeConfig;
    @FXML private Spinner<Integer> kValueSpinner;
    @FXML private ComboBox<String> distanceMetricComboBox;
    @FXML private Spinner<Double> learningRateSpinner;
    @FXML private Spinner<Integer> epochsSpinner;
    @FXML private Slider splitSlider;
    @FXML private Label splitLabel;
    @FXML private TextArea resultsArea;
    @FXML private GridPane confusionMatrixGrid;

    @FXML
    public void initialize() {
        // Initialize classifierComboBox with default selection
        classifierComboBox.getSelectionModel().selectedItemProperty().addListener(
                (observable, oldValue, newValue) -> updateModelConfiguration(newValue)
        );

        // Initialize spinners with value factories
        kValueSpinner.setValueFactory(
                new SpinnerValueFactory.IntegerSpinnerValueFactory(1, 20, 3)
        );

        learningRateSpinner.setValueFactory(
                new SpinnerValueFactory.DoubleSpinnerValueFactory(0.0001, 1.0, 0.01, 0.01)
        );

        epochsSpinner.setValueFactory(
                new SpinnerValueFactory.IntegerSpinnerValueFactory(1, 1000, 100)
        );

        // Initialize distance metric combo box
        distanceMetricComboBox.setValue("Euclidean");

        // Initialize split slider
        splitSlider.valueProperty().addListener((obs, oldVal, newVal) ->
                splitLabel.setText(String.format("%.0f%%", newVal.doubleValue()))
        );

        // Select default classifier
        classifierComboBox.getSelectionModel().selectFirst();
    }

    private void updateModelConfiguration(String modelName) {
        knnConfig.setManaged(false);
        logisticConfig.setManaged(false);
        treeConfig.setManaged(false);
        knnConfig.setVisible(false);
        logisticConfig.setVisible(false);
        treeConfig.setVisible(false);

        switch (modelName) {
            case "K-Nearest Neighbors":
                knnConfig.setManaged(true);
                knnConfig.setVisible(true);
                break;
            case "Logistic Regression":
                logisticConfig.setManaged(true);
                logisticConfig.setVisible(true);
                break;
            case "Decision Tree":
                treeConfig.setManaged(true);
                treeConfig.setVisible(true);
                break;
        }
    }

    @FXML
    private void handleBrowseButton() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Select Dataset");
        fileChooser.getExtensionFilters().add(
                new FileChooser.ExtensionFilter("CSV Files", "*.csv")
        );

        File file = fileChooser.showOpenDialog(null);
        if (file != null) {
            filePathField.setText(file.getAbsolutePath());
        }
    }

    private Model<Double, Integer> currentModel;
    private List<Instance<Double, Integer>> allData;

    @FXML
    private void handleTrainButton() {
        try {
            if (filePathField.getText().isEmpty() || classifierComboBox.getValue() == null) {
                showAlert("Error", "Please select both a file and a classifier.");
                return;
            }

            // Create a new model instance each time
            currentModel = createModel();
            String filePath = filePathField.getText();
            double splitRatio = splitSlider.getValue() / 100.0;

            // Load data if it's a new file or not loaded yet
            if (allData == null || !filePath.equals(lastLoadedFile)) {
                CSVConvert converter = new CSVConvert(filePath);
                allData = converter.getInstances();
                lastLoadedFile = filePath;
            }

            // Shuffle data for new split
            List<Instance<Double, Integer>> shuffledData = new ArrayList<>(allData);
            Collections.shuffle(shuffledData);

            // Split data
            int trainSize = (int) (shuffledData.size() * splitRatio);
            List<Instance<Double, Integer>> trainData = shuffledData.subList(0, trainSize);
            List<Instance<Double, Integer>> testData = shuffledData.subList(trainSize, shuffledData.size());

            // Train and test
            currentModel.train(trainData);
            List<Integer> predictions = currentModel.test(testData);
            List<Integer> actual = testData.stream()
                    .map(Instance::getOutput)
                    .collect(Collectors.toList());

            // Display actual results
            displayResults(predictions, actual);

        } catch (Exception e) {
            showAlert("Error", "An error occurred: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private String lastLoadedFile;

    private Model<Double, Integer> createModel() {
        DistanceMetric<Double> metric;
        String modelType = classifierComboBox.getValue();
        switch (modelType) {
            case "K-Nearest Neighbors":
                double[] minVals = new double[8];
                double[] maxVals = new double[8];
                Arrays.fill(minVals, Double.MAX_VALUE);
                Arrays.fill(maxVals, Double.MIN_VALUE);

                if (distanceMetricComboBox.getValue().equals("Euclidean"))
                    metric = new EuclideanDistance();
                else
                    metric = distanceMetricComboBox.getValue().equals("Manhattan") ?
                                                            new ManhattanDistance() : new DiabetesDistanceMetric(minVals, maxVals);
                return new KNNClassifier<>(kValueSpinner.getValue(), metric);

            case "Logistic Regression":
                return new LogisticRegression(
                        learningRateSpinner.getValue(),
                        epochsSpinner.getValue()
                );

            case "Decision Tree":
                return new DecisionTree<>();

            default:
                throw new IllegalStateException("Unknown model type: " + modelType);
        }
    }

    private void displayResults(List<Integer> predictions, List<Integer> actual) {
        int tp = 0, fp = 0, tn = 0, fn = 0;

        for (int i = 0; i < predictions.size(); i++) {
            if (predictions.get(i) == 1 && actual.get(i) == 1) tp++;
            else if (predictions.get(i) == 1 && actual.get(i) == 0) fp++;
            else if (predictions.get(i) == 0 && actual.get(i) == 0) tn++;
            else fn++;
        }

        double accuracy = (double) (tp + tn) / (tp + tn + fp + fn);
        double precision = tp > 0 ? (double) tp / (tp + fp) : 0.0;
        double recall = tp > 0 ? (double) tp / (tp + fn) : 0.0;
        double f1Score = precision + recall > 0 ?
                2 * (precision * recall) / (precision + recall) : 0.0;

        resultsArea.setText(String.format("""
            Model Evaluation Results:
            Accuracy: %.2f%%
            Precision: %.2f%%
            Recall: %.2f%%
            F1 Score: %.2f
            """,
                accuracy * 100,
                precision * 100,
                recall * 100,
                f1Score
        ));

        displayConfusionMatrix(new int[][]{
                {tn, fp},
                {fn, tp}
        });
    }

    private void displayConfusionMatrix(int[][] matrix) {
        confusionMatrixGrid.getChildren().clear();

        confusionMatrixGrid.add(new Label("Predicted →"), 0, 0);
        confusionMatrixGrid.add(new Label("Actual ↓"), 0, 1);
        confusionMatrixGrid.add(new Label("Negative"), 1, 0);
        confusionMatrixGrid.add(new Label("Positive"), 2, 0);

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                Label cell = new Label(String.valueOf(matrix[i][j]));
                cell.setStyle(
                        "-fx-padding: 10; " +
                                "-fx-border-color: black; " +
                                "-fx-background-color: #f8f9fa;"
                );
                confusionMatrixGrid.add(cell, j + 1, i + 1);
            }
        }
    }

    private void showAlert(String title, String content) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle(title);
        alert.setContentText(content);
        alert.showAndWait();
    }
}