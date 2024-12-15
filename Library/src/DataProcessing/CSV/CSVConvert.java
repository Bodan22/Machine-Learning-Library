package DataProcessing.CSV;

import DataProcessing.domain.Instance;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class CSVConvert {
    private final String fileName;
    private List<Instance<Double, Integer>> instances;

    public CSVConvert(String fileName) throws IOException {
        this.fileName = fileName;
        instances = new ArrayList<>();
        readFile();
    }

    public void readFile() throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            String line;

            while ((line = br.readLine()) != null) {

                String[] tokens = line.split(",");
                if (tokens.length != 9) {
                    throw new IOException("Invalid file format: expected 9 columns but got " + tokens.length);
                }

                try {
                    List<Double> input = new ArrayList<>();
                    for (int i = 0; i < 8; i++) {
                        input.add(Double.parseDouble(tokens[i].trim()));
                    }

                    // Parse the output (last column) correctly using parseInt
                    Integer output = Integer.parseInt(tokens[8].trim());

                    Instance<Double, Integer> instance = new Instance<>(input, output);
                    instances.add(instance);
                } catch (NumberFormatException e) {
                    System.err.println("Error parsing line: " + line);
                    throw new IOException("Error parsing numeric values: " + e.getMessage());
                }
            }
        } catch (Exception e) {
            throw new IOException("Error reading CSV file: " + e.getMessage());
        }
    }

    public List<Instance<Double, Integer>> getInstances() {
        return instances;
    }
}