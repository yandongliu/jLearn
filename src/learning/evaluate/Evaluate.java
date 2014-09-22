package learning.evaluate;

import java.util.List;

/**
 * Created by yandong on 7/16/14.
 */
public class Evaluate {

    /**
     * Compute RMSE and MAE
     * @param al1
     * @param al2
     */
    public static void computeError(List<Double> al1, List<Double> al2) {
        assert  al1.size() == al2.size();
        double mae = 0.0, rmse = 0.0; //mean absoltue error
        for(int i=0;i<al1.size();i++) {
            double d1 = al1.get(i);
            double d2 = al2.get(i);
            double dd = d1 - d2;
            mae += Math.abs(dd);
            rmse += dd*dd;
        }
        System.out.println("mae:"+mae/al1.size());
        System.out.println("rmse:"+Math.sqrt(rmse/al1.size()));
    }

    public static  void computeRegressionError(double[] al1, double[]  al2) {
        assert  al1.length == al2.length;
        double mae = 0.0, rmse = 0.0; //mean absoltue error
        for(int i=0;i<al1.length;i++) {
            double d1 = al1[i];
            double d2 = al2[i];
            double dd = d1 - d2;
            mae += Math.abs(dd);
            rmse += dd*dd;
        }
        System.out.println("mae:"+mae/al1.length);
        System.out.println("rmse:"+Math.sqrt(rmse/al1.length));
    }

    public static void computeClassificationError(String[] al1, String[]  al2) {
        assert  al1.length == al2.length;
        double mae = 0.0, rmse = 0.0; //mean absoltue error
        for(int i=0;i<al1.length;i++) {
            int dd = al1[i].equals(al2[i])?0:1;
            mae += dd;
            rmse += dd*dd;
        }
        System.out.println("mae:"+mae/al1.length);
        System.out.println("rmse:"+Math.sqrt(rmse/al1.length));
    }
}
