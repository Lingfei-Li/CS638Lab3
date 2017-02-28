package lingfei.CS638.Lab3.Utils;


import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

import java.awt.*;
import java.util.List;

public class PlotUtil extends ApplicationFrame{

    public PlotUtil(String applicationTitle) {
        super(applicationTitle);
    }

    public void plot(String chartTitle, List<Double> trainCurve, List<Double> tuneCurve) {
        JFreeChart lineChart = ChartFactory.createLineChart(
                chartTitle,
                "Epoch","Accuracy",
                createDataset(trainCurve, tuneCurve),
                PlotOrientation.VERTICAL,
                true,true,false);

        ChartPanel chartPanel = new ChartPanel( lineChart );
        chartPanel.setPreferredSize( new java.awt.Dimension( 560 , 367 ) );
        setContentPane( chartPanel );

        CategoryPlot plot = (CategoryPlot) lineChart.getPlot();
        plot.setRangePannable(true);
        plot.setRangeGridlinesVisible(true);
        plot.setBackgroundAlpha(1);
        plot.setBackgroundPaint(Color.white);
        plot.setRangeGridlinePaint(Color.gray);

        this.pack( );
        RefineryUtilities.centerFrameOnScreen( this );
        this.setVisible( true );
    }

    private DefaultCategoryDataset createDataset(List<Double> trainCurve, List<Double> tuneCurve) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset( );
        for(int i = 0; i < trainCurve.size(); i ++) {
            dataset.addValue( trainCurve.get(i), "Training Set", i+"" );
        }
        for(int i = 0; i < tuneCurve.size(); i ++) {
            dataset.addValue( tuneCurve.get(i), "Tuning Set", i+"" );
        }
        return dataset;
    }

}
