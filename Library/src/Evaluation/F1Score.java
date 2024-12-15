package Evaluation;

public class F1Score<F, L> { // F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    public static double calculate(double recall, double precision) {


        // Calculate F1 Score
        if (precision + recall == 0.0) {
            return 0.0;  // Handle case where both precision and recall are 0
        }

        return 2 * (precision * recall) / (precision + recall);
    }


    public static String formatAsPercentage(double f1Score) {
        return String.format("%.2f%%", f1Score * 100);
    }

}