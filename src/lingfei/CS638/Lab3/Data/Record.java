package lingfei.CS638.Lab3.Data;

public class Record {
    public double[][][] data;
    public int label;
    public Record(double[][][] data, double label) {
        this.data = data.clone();
        this.label = (int)label;
    }
}
