package Models;

import DataProcessing.domain.Instance;
import DataProcessing.domain.Model;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class LogisticRegression implements Model<Double, Integer> {
    private double[] weights;
    private double learningRate;
    private int epochs;
    private static final int NUM_FEATURES = 8;

    public LogisticRegression(double learningRate, int epochs) {
        // Initialize weights for all features plus bias
        this.weights = new double[NUM_FEATURES + 1];
        this.learningRate = learningRate;
        this.epochs = epochs;
    }

    // Normalize a feature column
    private double[] normalizeFeature(List<Double> feature) {
        double mean = feature.stream()
                .mapToDouble(Double::doubleValue)
                .average()
                .orElse(0.0);

        double std = Math.sqrt(feature.stream()
                .mapToDouble(x -> Math.pow(x - mean, 2))
                .average()
                .orElse(0.0));

        return feature.stream()
                .mapToDouble(x -> (x - mean) / (std == 0 ? 1 : std))
                .toArray();
    }


//    // Prepare and normalize the feature matrix
//    private double[][] prepareFeatures(double[][] rawData) {
//        double[][] normalizedData = new double[rawData.length][NUM_FEATURES];
//
//        // Normalize each feature column
//        for (int feature = 0; feature < NUM_FEATURES; feature++) {
//            double[] featureColumn = new double[rawData.length];
//            for (int row = 0; row < rawData.length; row++) {
//                featureColumn[row] = rawData[row][feature];
//            }
//            double[] normalizedFeature = normalizeFeature(featureColumn);
//            for (int row = 0; row < rawData.length; row++) {
//                normalizedData[row][feature] = normalizedFeature[row];
//            }
//        }
//
//        return normalizedData;
//    }

    private double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }


    @Override
    public void train(List<Instance<Double, Integer>> instances) {
        int numFeatures = instances.get(0).getInput().size();

        // Prepare normalized features
        List<List<Double>> features = new ArrayList<>();
        for (int i = 0; i < numFeatures; i++) {
            int featureIndex = i;
            List<Double> featureColumn = instances.stream()
                    .map(instance -> instance.getInput().get(featureIndex))
                    .collect(Collectors.toList());
            features.add(featureColumn);
        }

        // Normalize each feature
        double[][] normalizedFeatures = new double[numFeatures][];
        for (int i = 0; i < numFeatures; i++) {
            normalizedFeatures[i] = normalizeFeature(features.get(i));
        }

        // Training loop
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < instances.size(); i++) {
                // Calculate prediction
                double[] instanceFeatures = new double[numFeatures];
                for (int j = 0; j < numFeatures; j++) {
                    instanceFeatures[j] = normalizedFeatures[j][i];
                }

                double prediction = predict(instanceFeatures);
                double error = instances.get(i).getOutput() - prediction;

                // Update weights
                weights[0] += learningRate * error; // bias
                for (int j = 0; j < numFeatures; j++) {
                    weights[j + 1] += learningRate * error * instanceFeatures[j];
                }
            }
        }
    }

    @Override
    public List<Integer> test(List<Instance<Double, Integer>> instances) {
        List<Integer> predictions = new ArrayList<>();

        for (Instance<Double, Integer> instance : instances) {
            double[] features = instance.getInput().stream()
                    .mapToDouble(Double::doubleValue)
                    .toArray();
            predictions.add(predictClass(features));
        }

        return predictions;
    }

    private double predict(double[] features) {
        double z = weights[0]; // bias
        for (int i = 0; i < features.length; i++) {
            z += weights[i + 1] * features[i];
        }
        return sigmoid(z);
    }

    private int predictClass(double[] features) {
        return predict(features) >= 0.5 ? 1 : 0;
    }
}
