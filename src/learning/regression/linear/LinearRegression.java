package learning.regression.linear;

/**
 * Created by yandong on 7/11/14.
 */
import java.io.*;
import java.util.*;

public class LinearRegression {

    public LinearRegression(String fn, double alpha, double lambda, int ITER_MAX) {
        ArrayList<double[]> data = new ArrayList<double[]>();
        List<Integer> l_features_id = new ArrayList<Integer>();
        List<Double> data_targets = new ArrayList<Double>();

        int nfeatures = 0;

        try {
            BufferedReader br = new BufferedReader(new FileReader(fn));
            String[] aa = br.readLine().split(",");
            int label_col = -1;
            int ncols = aa.length;

            // features
            for (int i = 0; i < ncols; i++) {
                aa[i] = aa[i].trim();
                if (!aa[i].equals("label"))
                    l_features_id.add(i);
                else
                    label_col = i;
            }
            if (label_col == -1) {
                System.out.println("No label column!");
                System.exit(1);
            }

            nfeatures = l_features_id.size();
            String s;

            // read content into data, targets
            while ((s = br.readLine()) != null) {
                aa = s.split(",");
                double[] dd = new double[nfeatures];
                for (int i = 0; i < nfeatures; i++) {
                    if (!aa[l_features_id.get(i)].equals("?"))
                        dd[i] = Double.parseDouble(aa[l_features_id.get(i)]);
                }
                data.add(dd);
                data_targets.add(Double.parseDouble(aa[label_col]));
            }

            br.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        ArrayList<double[]> data2 = normalize(data, nfeatures);
        double[] target_stats = normalizeTarget(data_targets);
        double target_mean = target_stats[0];
        double target_stderr = target_stats[1];

        // long seed = System.nanoTime();
        // Collections.shuffle(data, new Random(seed));

        // gradient descent for thetas
        double[] thetas = optimize(data2, l_features_id, nfeatures, data_targets, alpha, lambda, ITER_MAX);

        // l_targets = new ArrayList<String>();
        //
        // HashMap<String, double[]> mThetas =
        // readLiblinearParams("/home/yandong/Dropbox/code/node/lr/leaf.libsvm.model",
        // l_targets);
        // ntargets = l_targets.size();
        printThetas(thetas);

        double diff = 0;
        for (int i = 0; i < data.size(); i++) {
            double sample_target = data_targets.get(i);
            double predict = predict(thetas, data.get(i), nfeatures);
            // System.out.println("real:" + sample_target + " predicted:"
            // + predict);
            predict= predict*target_stderr+target_mean;
            double delta = sample_target - predict;
            diff += delta * delta;
        }
        System.out.println("MSE:" + diff / data.size());
    }

    double[] normalizeTarget(List<Double> data) {
        int n = data.size();
        double[] r = new double[2];
        double mean = 0.0, stderr = 0.0;
        for (int i = 0; i < n; i++) {
            mean += data.get(i);
        }
        mean /= n;
        for (int i = 0; i < n; i++) {
            double a = data.get(i) - mean;
            stderr += a * a;
        }
        stderr = Math.sqrt(stderr / n);
        // System.out.println(mean+" "+variance);
        for (int i = 0; i < n; i++) {
            data.set(i, (data.get(i)- mean)/stderr);
        }
        r[0] = mean;
        r[1] = stderr;
        return r;
    }

    ArrayList<double[]> normalize(List<double[]> data, int nfeatures) {
        ArrayList<double[]> data2 = new ArrayList<double[]>();
        for (int i = 0; i < data.size(); i++) {
            double mean = 0.0, stderr = 0.0;
            for (int j = 0; j < nfeatures; j++) {
                mean += data.get(i)[j];
            }
            mean /= nfeatures;
            for (int j = 0; j < nfeatures; j++) {
                double a = data.get(i)[j] - mean;
                stderr += a * a;
            }
            stderr = Math.sqrt(stderr / nfeatures);
            // System.out.println(mean+" "+variance);
            double[] dd = new double[nfeatures];
            for (int j = 0; j < nfeatures; j++) {
                dd[j] = (data.get(i)[j] - mean)/stderr;
            }
            data2.add(dd);
        }
        return data2;
    }

    double predict(double[] theta, double[] sample_data, int nfeatures) {
        return compThetaXProdct(theta, sample_data, nfeatures);
    }

    void printThetas(double[] thetas) {
        System.out.println("printThetas");
        for (int i = 0; i < thetas.length; i++) {
            System.out.print(thetas[i] + " ");
        }
        System.out.println();
    }

    void gd_batch(double[] theta, ArrayList<double[]> data, List<Integer> l_features, int nfeatures,
                  List<Double> data_targets, double alpha, double lambda) {
        double[] gradient = new double[nfeatures];

        for (int i = 0; i < data.size(); i++) {
            double[] sample_data = data.get(i);
            double prdt = compThetaXProdct(theta, sample_data, nfeatures);
            for (int k = 0; k < nfeatures; k++) {
                gradient[k] += ((data_targets.get(i) - prdt) * sample_data[k]);
            }
        }

        for (int k = 0; k < nfeatures; k++) {
            System.out.println(gradient[k]);
            theta[k] += (alpha * gradient[k] - 2 * data.size() * lambda * theta[k]);
        }
    }

    void sgd_once(double[] theta, ArrayList<double[]> data, List<Integer> l_features, int nfeatures,
                  List<Double> data_targets, double alpha, double lambda) {
        for (int i = 0; i < data.size(); i++) {
            double[] sample_data = data.get(i);
            double prdt = compThetaXProdct(theta, sample_data, nfeatures);
            for (int k = 0; k < nfeatures; k++)
                theta[k] += (alpha * (data_targets.get(i) - prdt) * sample_data[k] - 2 * lambda * theta[k]);
            theta[nfeatures] += (alpha * (data_targets.get(i) - prdt) - 2 * lambda * theta[nfeatures]);
        }
    }

    /***
     * Optimize by batch gradient descent or SGD
     */
    double[] optimize(ArrayList<double[]> data, List<Integer> l_features, int nfeatures, List<Double> data_targets,
                      double alpha, double lambda, int ITER_MAX) {
        double[] theta = new double[nfeatures + 1]; // last one is for x===1
        for (int i = 0; i <= nfeatures; i++) {
            theta[i] = 1.0;
        }

        for (int i = 0; i < ITER_MAX; i++) {
            sgd_once(theta, data, l_features, nfeatures, data_targets, alpha, lambda);
            // System.out.println("Iteration " + i + "...");
            // gd_batch(theta, data, l_features, nfeatures, targets,alpha,
            // lambda);
            // printThetas(theta);
        }
        return theta;
    }

    double compThetaXProdct(double[] theta, double[] feaVals, int n) {

        double prdt = 0.0;
        for (int i = 0; i < n; i++)
            prdt += (theta[i] * feaVals[i]);
        prdt += theta[n];
        return prdt;
    }

    public static void main(String[] args) {
        // new LR(args[0], args[1], Double.parseDouble(args[2]),
        // Integer.parseInt(args[3]));
        long t1 = System.currentTimeMillis();
        new LinearRegression("/home/yandong/Dropbox/code/node/lr/autoPrice.csv", 0.005, 0.000000, 100);
        long t2 = System.currentTimeMillis();
        System.out.println(t2 - t1);
        // MSE:1.2689226982212342E7 1000 with x===1
        // MSE:1.234941844299759E7 1000 without x===1
        // weka: Mean absolute error 1644.0143

    }
}
