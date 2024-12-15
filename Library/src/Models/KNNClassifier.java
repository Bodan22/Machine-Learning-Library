package Models;

import DataProcessing.domain.Instance;
import DataProcessing.domain.Model;
import utils.DistanceMetric;

import java.util.*;

public class KNNClassifier<F, L> implements Model<F, L> {
    private List<Instance<F, L>> trainingData;
    private final int k;
    private final DistanceMetric<F> distanceMetric;

    public KNNClassifier(int k, DistanceMetric<F> distanceMetric) {
        if (k <= 0) {
            throw new IllegalArgumentException("K must be greater than 0");
        }
        this.k = k;
        this.distanceMetric = distanceMetric;
        this.trainingData = new ArrayList<>();
    }

    @Override
    public void train(List<Instance<F, L>> instances) {
        if (instances == null || instances.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be null or empty");
        }
        this.trainingData = new ArrayList<>(instances);
    }

    @Override
    public List<L> test(List<Instance<F, L>> instances) {
        if (instances == null || instances.isEmpty()) {
            throw new IllegalArgumentException("Test instances cannot be null or empty");
        }

        List<L> predictions = new ArrayList<>();
        for (Instance<F, L> instance : instances) {
            predictions.add(predict(instance.getInput()));
        }
        return predictions;
    }

    private L predict(List<F> features) {
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("Features cannot be null or empty");
        }

        List<DistanceLabel> distances = new ArrayList<>();

        // Calculate distances to all training instances
        for (Instance<F, L> trainInstance : trainingData) {
            double distance = distanceMetric.calculate(features, trainInstance.getInput());
            distances.add(new DistanceLabel(distance, trainInstance.getOutput()));
        }

        // Sort by distance
        Collections.sort(distances);

        // Get k nearest neighbors
        Map<L, Integer> labelCounts = new HashMap<>();
        for (int i = 0; i < k && i < distances.size(); i++) {
            L label = distances.get(i).label;
            labelCounts.merge(label, 1, Integer::sum);
        }

        // Return majority label
        return labelCounts.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .orElseThrow(() -> new RuntimeException("No majority label found"))
                .getKey();
    }

    private class DistanceLabel implements Comparable<DistanceLabel> {
        final double distance;
        final L label;

        DistanceLabel(double distance, L label) {
            this.distance = distance;
            this.label = label;
        }

        @Override
        public int compareTo(DistanceLabel other) {
            return Double.compare(this.distance, other.distance);
        }
    }
}