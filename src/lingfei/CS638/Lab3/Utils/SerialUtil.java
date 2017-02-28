package lingfei.CS638.Lab3.Utils;

import lingfei.CS638.Lab3.CNN.CNN;

import java.io.FileNotFoundException;
import java.io.PrintStream;

public class SerialUtil {

    private CNN model;
    private String defaultFileName = "CNN_model.data";

    public SerialUtil(CNN model) {
        this.model = model;
    }

    public void saveModelToFile() { saveModelToFile(defaultFileName); }

    public void loadModelFromFile() { loadModelFromFile(defaultFileName); }


    public void saveModelToFile(String filename) {
        try(PrintStream ps = new PrintStream(filename)) {


            ps.println("");

        } catch (FileNotFoundException e) {
            System.err.println("File not found when saving CNN model");
        }
    }

    public void loadModelFromFile(String filename) {

    }





}
