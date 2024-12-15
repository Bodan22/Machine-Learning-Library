import DataProcessing.CSV.CSVConvert;
import DataProcessing.domain.Instance;
import Evaluation.Accuracy;
import Evaluation.Precision;
import Models.DecisionTree;
import Models.KNNClassifier;
import Models.LogisticRegression;
import utils.DiabetesDistanceMetric;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class ClassifierComparison {
    public static void main(String[] args) throws IOException {
        // Load the data
        CSVConvert converter = new CSVConvert("D:\\MAP\\lb-Bodan22\\lb\\src\\diabetes.csv");
        List<Instance<Double, Integer>> allInstances = new ArrayList<>(converter.getInstances());

        // Shuffle the data
        Collections.shuffle(allInstances);

        // Split data into training and testing sets
        int trainSize = (int) (0.8 * allInstances.size());
        List<Instance<Double, Integer>> trainData = new ArrayList<>();
        List<Instance<Double, Integer>> testData = new ArrayList<>();

        // Properly split the data
        for (int i = 0; i < allInstances.size(); i++) {
            Instance<Double, Integer> instance = allInstances.get(i);
            if (i < trainSize) {
                trainData.add(instance);
            } else {
                testData.add(instance);
            }
        }

        // Verify data integrity
        verifyDataIntegrity(trainData, "Training");
        verifyDataIntegrity(testData, "Testing");

        // Test all classifiers
        System.out.println("Testing classifiers on diabetes dataset...\n");

        // 1. Test KNN
        System.out.println("1. K-Nearest Neighbors Classifier");
        testKNN(trainData, testData);

        // 2. Test Decision Tree
        System.out.println("\n2. Decision Tree Classifier");
        testDecisionTree(trainData, testData);

        // 3. Test Logistic Regression
        System.out.println("\n3. Logistic Regression Classifier");
        testLogisticRegression(trainData, testData);
    }

    private static void verifyDataIntegrity(List<Instance<Double, Integer>> data, String datasetName) {
        System.out.println("\nVerifying " + datasetName + " data integrity:");
        System.out.println("Dataset size: " + data.size());

        boolean hasNullInputs = false;
        boolean hasNullOutputs = false;

        for (int i = 0; i < data.size(); i++) {
            Instance<Double, Integer> instance = data.get(i);
            if (instance.getInput() == null || instance.getInput().isEmpty()) {
                hasNullInputs = true;
                System.out.println("Null/empty input at index " + i);
            }
            if (instance.getOutput() == null) {
                hasNullOutputs = true;
                System.out.println("Null output at index " + i);
            }
        }

        if (!hasNullInputs && !hasNullOutputs) {
            System.out.println("Data integrity check passed");
        }
    }

    private static void testKNN(List<Instance<Double, Integer>> trainData,
                                List<Instance<Double, Integer>> testData) {
        try {
            // Calculate min and max values for normalization
            double[] minVals = new double[8];
            double[] maxVals = new double[8];
            Arrays.fill(minVals, Double.MAX_VALUE);
            Arrays.fill(maxVals, Double.MIN_VALUE);

            for (Instance<Double, Integer> instance : trainData) {
                List<Double> features = instance.getInput();
                for (int i = 0; i < 8; i++) {
                    minVals[i] = Math.min(minVals[i], features.get(i));
                    maxVals[i] = Math.max(maxVals[i], features.get(i));
                }
            }

            // Create and train KNN
            DiabetesDistanceMetric distanceMetric = new DiabetesDistanceMetric(minVals, maxVals);
            KNNClassifier<Double, Integer> knn = new KNNClassifier<>(5, distanceMetric);

            long startTime = System.currentTimeMillis();
            knn.train(trainData);
            List<Integer> predictions = knn.test(testData);
            long endTime = System.currentTimeMillis();

            // Calculate metrics
            Accuracy<Double, Integer> accuracy = new Accuracy<>();
            Precision<Double, Integer> precision = new Precision<>();

            System.out.printf("Accuracy: %.2f%%\n", accuracy.evaluate(testData, predictions) * 100);
            System.out.printf("Precision: %.2f%%\n", precision.evaluate(testData, predictions) * 100);
            System.out.printf("Execution time: %d ms\n", endTime - startTime);

        } catch (Exception e) {
            System.out.println("Error in KNN evaluation: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void testDecisionTree(List<Instance<Double, Integer>> trainData,
                                         List<Instance<Double, Integer>> testData) {
        try {
            DecisionTree<Double, Integer> dt = new DecisionTree<>();

            long startTime = System.currentTimeMillis();
            dt.train(trainData);
            List<Integer> predictions = dt.test(testData);
            long endTime = System.currentTimeMillis();

            Accuracy<Double, Integer> accuracy = new Accuracy<>();
            Precision<Double, Integer> precision = new Precision<>();

            System.out.printf("Accuracy: %.2f%%\n", accuracy.evaluate(testData, predictions) * 100);
            System.out.printf("Precision: %.2f%%\n", precision.evaluate(testData, predictions) * 100);
            System.out.printf("Execution time: %d ms\n", endTime - startTime);

        } catch (Exception e) {
            System.out.println("Error in Decision Tree evaluation: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void testLogisticRegression(List<Instance<Double, Integer>> trainData,
                                               List<Instance<Double, Integer>> testData) {
        try {
            LogisticRegression lr = new LogisticRegression(0.01, 100);

            long startTime = System.currentTimeMillis();
            lr.train(trainData);
            List<Integer> predictions = lr.test(testData);
            long endTime = System.currentTimeMillis();

            Accuracy<Double, Integer> accuracy = new Accuracy<>();
            Precision<Double, Integer> precision = new Precision<>();

            System.out.printf("Accuracy: %.2f%%\n", accuracy.evaluate(testData, predictions) * 100);
            System.out.printf("Precision: %.2f%%\n", precision.evaluate(testData, predictions) * 100);
            System.out.printf("Execution time: %d ms\n", endTime - startTime);

        } catch (Exception e) {
            System.out.println("Error in Logistic Regression evaluation: " + e.getMessage());
            e.printStackTrace();
        }
    }
}