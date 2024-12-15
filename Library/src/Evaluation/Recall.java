package Evaluation;

import java.util.List;

public class Recall {
    public static double calculate(List<Integer> predictions, List<Integer> actual) {
        if (predictions == null || actual == null) {
            throw new IllegalArgumentException("Input lists cannot be null");
        }

        if (predictions.size() != actual.size()) {
            throw new IllegalArgumentException("Predictions and actual values must have the same size");
        }

        if (predictions.isEmpty()) {
            throw new IllegalArgumentException("Input lists cannot be empty");
        }

        int truePositives = 0;
        int falseNegatives = 0;

        // Count true positives and false negatives
        for (int i = 0; i < predictions.size(); i++) {
            int predicted = predictions.get(i);
            int actualValue = actual.get(i);

            // Validate input values
            if (predicted != 0 && predicted != 1) {
                throw new IllegalArgumentException("Prediction values must be 0 or 1");
            }
            if (actualValue != 0 && actualValue != 1) {
                throw new IllegalArgumentException("Actual values must be 0 or 1");
            }

            if (actualValue == 1) {  // Actual positive case
                if (predicted == 1) {
                    truePositives++;  // Correctly predicted positive
                } else {
                    falseNegatives++;  // Missed positive case
                }
            }
        }

        // Avoid division by zero
        if (truePositives + falseNegatives == 0) {
            return 0.0;  // No positive cases in the actual values
        }

        return (double) truePositives / (truePositives + falseNegatives);
    }

    public static String formatAsPercentage(double recall) {
        return String.format("%.2f%%", recall * 100);
    }
}