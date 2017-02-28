package lingfei.CS638.Lab3.Utils;

public class Size {
    public int x;
    public int y;
    public Size(int x, int y) {
        this.x = x;
        this.y = y;
    }
    public Size plus(int diff) { return new Size(x + diff, y + diff); }
    public Size divide(int divider) { return new Size(x / divider, y / divider); }
    public Size minus(Size diff) { return new Size(x - diff.x, y - diff.y); }
    public boolean equals(final Size rhs) { return this.x == rhs.x && this.y == rhs.y; }
    public String toString() { return x + " " + y; }
}
