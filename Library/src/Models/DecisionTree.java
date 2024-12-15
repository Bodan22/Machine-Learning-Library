package Models;


import DataProcessing.CSV.CSVConvert;
import DataProcessing.domain.Instance;
import DataProcessing.domain.Model;

import java.util.*;
import java.util.stream.Collectors;

public class DecisionTree<F, L> implements Model<F, L> {
    private Node root;
    private CSVConvert CSV;


    @Override
    public void train(List<Instance<F, L>> instances) {
        // Implementation of decision tree training
        List<List<F>> inputs = instances.stream()
                .map(Instance::getInput)
                .collect(Collectors.toList());
        List<L> outputs = instances.stream()
                .map(Instance::getOutput)
                .collect(Collectors.toList());

        this.root = buildTree(inputs, outputs, 0);
    }

    @Override
    public List<L> test(List<Instance<F, L>> instances) {
        return instances.stream()
                .map(instance -> predict(instance.getInput()))
                .collect(Collectors.toList());
    }

    private L predict(List<F> input) {
        // Implementation of prediction logic
        return traverseTree(input, root);
    }

    private Node buildTree(List<List<F>> inputs, List<L> outputs, int depth) {
        List<List<Double>> doubleFeatures = convertToDouble(inputs);
        List<Integer> intLabels = convertToInteger(outputs);

        // Find the best split
        SplitInfo bestSplit = findBestSplit(doubleFeatures, intLabels);

        // If no good split found, create leaf node
        if (bestSplit.gainValue <= 0) {
            return new Node(getMajorityClass(outputs));
        }

        // Split the data
        List<Integer> leftIndices = new ArrayList<>();
        List<Integer> rightIndices = new ArrayList<>();

        for (int i = 0; i < inputs.size(); i++) {
            double value = ((Number) inputs.get(i).get(bestSplit.feature)).doubleValue();
            if (value < bestSplit.splitValue) {
                leftIndices.add(i);
            } else {
                rightIndices.add(i);
            }
        }

        // Create child nodes
        List<List<F>> leftInputs = getSubset(inputs, leftIndices);
        List<L> leftOutputs = getSubset(outputs, leftIndices);
        List<List<F>> rightInputs = getSubset(inputs, rightIndices);
        List<L> rightOutputs = getSubset(outputs, rightIndices);

        Node leftChild = buildTree(leftInputs, leftOutputs, depth + 1);
        Node rightChild = buildTree(rightInputs, rightOutputs, depth + 1);

        @SuppressWarnings("unchecked")
        F splitValue = (F) Double.valueOf(bestSplit.splitValue);
        return new Node(splitValue, bestSplit.feature, leftChild, rightChild);
    }

    private L traverseTree(List<F> input, Node node) {
        // Step 1: Base case - if we've reached a leaf node
        if (node.prediction != null) {
            return node.prediction;
        }

        // Step 2: Get the feature value we need to check for this node
        F featureValue = input.get(node.splitFeature);

        // Step 3: Compare the feature value with the node's split value
        // We need to cast to make the comparison
        // Note: This assumes F is a comparable type like Double
        double inputValue = ((Number) featureValue).doubleValue();
        double splitValue = ((Number) node.splitValue).doubleValue();

        // Step 4: Traverse left or right based on comparison
        if (inputValue < splitValue) {
            return traverseTree(input, node.left);
        } else {
            return traverseTree(input, node.right);
        }
    }

    private SplitInfo findBestSplit(List<List<Double>> features, List<Integer> labels) {
        SplitInfo bestSplit = new SplitInfo(-1, 0.0, Double.NEGATIVE_INFINITY);

        // For each feature
        for (int feature = 0; feature < features.get(0).size(); feature++) {
            // Get unique values for the feature
            Set<Double> uniqueValues = new TreeSet<>();
            for (List<Double> instance : features) {
                uniqueValues.add(instance.get(feature));
            }

            // Try each potential split point
            Double[] values = uniqueValues.toArray(new Double[0]);
            for (int i = 0; i < values.length - 1; i++) {
                // Calculate split point as average between consecutive unique values
                double splitValue = (values[i] + values[i + 1]) / 2.0;

                // Calculate information gain for this split
                double gain = calculateInformationGain(features, labels, feature, splitValue);

                // Update best split if this is better
                if (gain > bestSplit.gainValue) {
                    bestSplit = new SplitInfo(feature, splitValue, gain);
                }
            }
        }

        return bestSplit;
    }

    // Calculate information gain for a potential split
    private double calculateInformationGain(List<List<Double>> features, List<Integer> labels,
                                            int feature, double splitValue) {
        // Calculate parent entropy
        double parentEntropy = calculateEntropy(labels);

        // Split the data
        List<Integer> leftLabels = new ArrayList<>();
        List<Integer> rightLabels = new ArrayList<>();

        for (int i = 0; i < features.size(); i++) {
            if (features.get(i).get(feature) < splitValue) {
                leftLabels.add(labels.get(i));
            } else {
                rightLabels.add(labels.get(i));
            }
        }

        // Calculate weighted average entropy of children
        double leftWeight = (double) leftLabels.size() / labels.size();
        double rightWeight = (double) rightLabels.size() / labels.size();

        double leftEntropy = calculateEntropy(leftLabels);
        double rightEntropy = calculateEntropy(rightLabels);

        // Calculate information gain
        return parentEntropy - (leftWeight * leftEntropy + rightWeight * rightEntropy);
    }

    // Calculate entropy for a set of labels
    private double calculateEntropy(List<Integer> labels) {
        if (labels.isEmpty()) return 0.0;

        // Count occurrences of each class
        Map<Integer, Integer> counts = new HashMap<>();
        for (Integer label : labels) {
            counts.put(label, counts.getOrDefault(label, 0) + 1);
        }

        // Calculate entropy
        double entropy = 0.0;
        int totalCount = labels.size();

        for (int count : counts.values()) {
            double probability = (double) count / totalCount;
            entropy -= probability * log2(probability);
        }

        return entropy;
    }

    private L getMajorityClass(List<L> outputs) {
        if (outputs.isEmpty()) return null;

        Map<L, Long> counts = outputs.stream()
                .collect(Collectors.groupingBy(o -> o, Collectors.counting()));

        return counts.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(outputs.get(0));
    }

    // Helper method to calculate log base 2
    private double log2(double x) {
        return Math.log(x) / Math.log(2);
    }

    @SuppressWarnings("unchecked")
    private List<List<Double>> convertToDouble(List<List<F>> inputs) {
        return inputs.stream()
                .map(list -> list.stream()
                        .map(value -> ((Number) value).doubleValue())
                        .collect(Collectors.toList()))
                .collect(Collectors.toList());
    }

    @SuppressWarnings("unchecked")
    private List<Integer> convertToInteger(List<L> outputs) {
        return outputs.stream()
                .map(value -> ((Number) value).intValue())
                .collect(Collectors.toList());
    }

    private <T> List<T> getSubset(List<T> list, List<Integer> indices) {
        return indices.stream()
                .map(list::get)
                .collect(Collectors.toList());
    }

    private class Node {
        F splitValue;
        int splitFeature;
        Node left;
        Node right;
        L prediction;

        // Constructor for decision nodes
        Node(F splitValue, int splitFeature, Node left, Node right) {
            this.splitValue = splitValue;
            this.splitFeature = splitFeature;
            this.left = left;
            this.right = right;
        }

        // Constructor for leaf nodes
        Node(L prediction) {
            this.prediction = prediction;
        }
    }

    private static class SplitInfo {
        int feature;
        double splitValue;
        double gainValue;

        SplitInfo(int feature, double splitValue, double gainValue) {
            this.feature = feature;
            this.splitValue = splitValue;
            this.gainValue = gainValue;
        }
    }

}