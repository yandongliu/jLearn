package learning.regression.boosting;

import learning.data.Data;
import learning.data.Data;

/**
 * Created by yandong on 7/15/14.
 */
enum LOSS_FUNC {
    REGRESSION
}
public class Loss {
    void compGradient(Data data, double[] predictions, double[] gradient) {
//        System.out.println("Loss::compGradient");
        shrinkage = 1.0;
    }
    void postGradient(Data data, double[] predictions, double[] gradient) {
//        System.out.println("Loss::postGradient");
    }
    int nTrees;
    LOSS_FUNC loss_type;
    public double mse, bias, shrinkage=1.0;
    public double dataSamplingRate =0.5, featureSamplingRate = 0.5;
    public double stepsize = 1.0;
}
