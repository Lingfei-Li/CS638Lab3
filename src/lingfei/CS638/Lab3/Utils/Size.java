package lingfei.CS638.Lab3.Utils;

public class Size {
    public int x;
    public int y;
    public Size(int x, int y) {
        this.x = x;
        this.y = y;
    }
    public Size plus(int diff) {
        x += diff;
        y += diff;
        return this;
    }
    public Size minus(Size diff) {
        x -= diff.x;
        y -= diff.y;
        return this;
    }

    public boolean equals(final Size rhs) {
        return this.x == rhs.x && this.y == rhs.y;
    }
}
