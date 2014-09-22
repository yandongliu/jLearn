package learning.linear;

/**
 * Created by yandong on 7/11/14.
 */
import learning.data.Data;
import learning.evaluate.Evaluate;
import learning.io.DataReader;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class LogisticRegression {
    /**
     * read the liblinear model file
     * @param fn
     * @param l_targets
     * @return
     */
    HashMap<String, double[]> readLiblinearParams(String fn, List<String> l_targets) {
        HashMap<String, double[]> mThetas = new HashMap<String, double[]>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(fn));
            String s;
            boolean inW = false;
            int nr_feature = 0;
            int feature_idx = 0;
            while ((s = br.readLine()) != null) {
                if (inW) {
                    String[] ss = s.split(" ");
                    for (int i = 0; i < l_targets.size(); i++) {
                        mThetas.get(l_targets.get(i))[feature_idx] = Double.parseDouble(ss[i]);
                    }
                    feature_idx++;
                    continue;
                } else if (s.startsWith("nr_feature")) {
                    String[] ss = s.split(" ");
                    nr_feature = Integer.parseInt(ss[1]);
                    for (String label : l_targets) {
                        double[] thetas = new double[nr_feature];
                        mThetas.put(label, thetas);
                    }
                } else if (s.startsWith("label")) {
                    String[] ss = s.split(" ");
                    for (int i = 1; i < ss.length; i++) {
                        l_targets.add(ss[i]);
                    }
                    continue;
                } else if (s.equals("w")) {
                    inW = true;
                    continue;
                }
            }
            br.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        return mThetas;
    }



    public LogisticRegression(String fn, String fn_test, double alpha, double lambda, int ITER_MAX) {
//        ArrayList<double[]> data = new ArrayList<double[]>();
//        List<Integer> l_features = new ArrayList<Integer>();
//        List<String> targets = new ArrayList<String>();
//
//        List<String> l_targets = new ArrayList<String>();
//        HashSet<String> s_targets = new HashSet<String>();
//        int ntargets = 0, nfeatures = 0;

        Data data = null;

        try {
            data = DataReader.readRealDataFile(fn);
        } catch (IOException ex) {
            ex.printStackTrace();
        }


        normalize(data.data, data.nFeatures);

//		long seed = System.nanoTime();
        // Collections.shuffle(data, new Random(seed));


        HashMap<String, double[]> mThetas = optimize(data.data, data.l_featureNames, data.nFeatures, data.targets, data.l_targets, data.nTargets,
                alpha, lambda, ITER_MAX);

        printThetas(data.l_targets, data.nTargets, mThetas);

//        ArrayList<double[]> data_test = new ArrayList<double[]>();
//        List<Integer> l_features_test = new ArrayList<Integer>();
//        List<String> data_targets_test = new ArrayList<String>();
//
//        HashSet<String> s_targets_test = new HashSet<String>();

        Data<String> testData = null;

        try {
            testData = DataReader.readRealDataFile(fn_test);
        } catch (IOException ex) {
            ex.printStackTrace();
        }

        normalize(testData.data, testData.nFeatures);

        int correct = 0;
        for (int i = 0; i < testData.data.size(); i++) {
            String sample_target = testData.targets.get(i);
            String predict = classify(mThetas, testData.data.get(i), testData.nFeatures, testData.l_targets, testData.nTargets);
            System.out.println("real:" + sample_target + " predicted:"+ predict);
//            String predict = "";
            if (sample_target.equals(predict)) {
                correct++;
            }
        }
        System.out.println("accuracy:" + (float) correct / testData.data.size() +" got "+correct+" out of "+ testData.data.size());
    }

    /**
     * normalize feature values
     * @param data
     * @param nfeatures
     */
    void normalize(List<double[]> data, int nfeatures) {
        for(int i=0;i<data.size();i++) {
            double mean=0.0, stderr = 0.0;
            for(int j=0;j<nfeatures;j++) {
                mean += data.get(i)[j];
            }
            mean/=nfeatures;
            for(int j=0;j<nfeatures;j++) {
                double a = data.get(i)[j]-mean;
                stderr += a*a;
            }
            stderr = Math.sqrt(stderr/nfeatures);
//			System.out.println(mean+" "+variance);
            for(int j=0;j<nfeatures;j++) {
                data.get(i)[j] -= mean;
                data.get(i)[j] /=stderr;
            }
        }
    }

    String classify(HashMap<String, double[]> mThetas, double[] sample_data, int nfeatures, List<String> l_targets,
                    int ntargets) {
        String max_target = l_targets.get(0);
        double max_p = compThetaXProdct(mThetas.get(l_targets.get(0)), sample_data, nfeatures);
        for (int i = 1; i < ntargets; i++) {
            String target = l_targets.get(i);
            double[] thetas = mThetas.get(target);
            double prdt = compThetaXProdct(thetas, sample_data, nfeatures);
            // System.out.println("class " + i + " " + prdt);
            if (max_p < prdt) {
                max_p = prdt;
                max_target = target;
            }
        }
        return max_target;
    }

    void printThetas(List<String> l_targets, int ntargets, HashMap<String, double[]> mThetas) {
        System.out.println("printThetas");
        for (int j = 0; j < ntargets; j++) {
            String target = l_targets.get(j);
            double[] thetas = mThetas.get(target);
            System.out.println("target:" + target);
            for (int i = 0; i < thetas.length; i++) {
                System.out.print(thetas[i] + " ");
            }
            System.out.println();
        }
    }

    void gd_batch(HashMap<String, double[]> mThetas, ArrayList<double[]> data, List<Integer> l_features, int nfeatures,
                  List<String> data_targets, List<String> l_targets, int ntargets, double alpha, double lambda) {
        for (int j = 0; j < ntargets; j++) {
            String target = l_targets.get(j);
            double[] gradient = new double[nfeatures];

            for (int i = 0; i < data.size(); i++) {
                double max_prdt = Integer.MIN_VALUE;
                double[] prdt = new double[ntargets];
                double this_prdt = 0.0;
                double[] sample_data = data.get(i);
                String sample_target = data_targets.get(i);

                // (theta . X) for all targets
                for (int k = 0; k < ntargets; k++) {
                    String target1 = l_targets.get(k);
                    double prdt1 = compThetaXProdct(mThetas.get(target1), sample_data, nfeatures);
                    if (j==k)
                        this_prdt = prdt1;
                    if (max_prdt < prdt1)
                        max_prdt = prdt1;
                    prdt[k] = prdt1;
                }
                double z = 0.0;
                for (int k = 0; k < ntargets; k++) {
                    z += (Math.exp(prdt[k] - max_prdt));
                }
                double p = Math.exp(this_prdt - max_prdt) / z;
                for (int k = 0; k < nfeatures; k++) {
                    if (sample_target.equals(target)) {
                        gradient[k] += ((1.0 - p) * sample_data[k]);
                    } else {
                        gradient[k] += ((0.0 - p) * sample_data[k]);
                    }
                }
            }
            double[] thetas = mThetas.get(target);
            for (int k = 0; k < nfeatures; k++) {
                thetas[k] += (alpha * gradient[k] - 2 * data.size() * lambda * thetas[k]);
            }
        }
    }

    void sgd_once(HashMap<String, double[]> mThetas, ArrayList<double[]> data, List<Integer> l_features, int nfeatures,
                  List<String> data_targets, List<String> l_targets, int ntargets, double alpha, double lambda) {
        for (int i = 0; i < data.size(); i++) {
            double[] prdt = new double[ntargets];
            double[] sample_data = data.get(i);
            String sample_target = data_targets.get(i);
            double max_prdt = prdt[0]= compThetaXProdct(mThetas.get(l_targets.get(0)), sample_data, nfeatures);
            for (int j = 1; j < ntargets; j++) {
                String target = l_targets.get(j);
                double prdt1 = compThetaXProdct(mThetas.get(target), sample_data, nfeatures);
                if (max_prdt < prdt1)
                    max_prdt = prdt1;
                prdt[j] = prdt1;
            }
            double z = 0.0;
            for (int j = 0; j < ntargets; j++) {
                z += (Math.exp(prdt[j] - max_prdt));
            }
            for (int j = 0; j < ntargets; j++) {
                double p = Math.exp(prdt[j] - max_prdt) / z;
                String target = l_targets.get(j);
//				System.out.println(target+":"+p);
                double[] thetas = mThetas.get(target);
                for (int k = 0; k < nfeatures; k++) {
                    if (sample_target.equals(target)) {
                        thetas[k] += (alpha * (1.0 - p) * sample_data[k] - 2 * lambda * thetas[k]);
                    } else {
                        thetas[k] += (alpha * (0.0 - p) * sample_data[k] - 2 * lambda * thetas[k]);
                    }
                }
                if (sample_target.equals(target)) {
//					thetas[nfeatures] += (alpha * (1.0 - p)  - 2 * lambda * thetas[nfeatures]);
                    thetas[nfeatures] += alpha * (1.0 - p);
                } else {
//					thetas[nfeatures] += (alpha * (0.0 - p)  - 2 * lambda * thetas[nfeatures]);
                    thetas[nfeatures] += alpha * (0.0 - p);
                }
            }
        }
    }

    /**
     * Optimize by batch gradient descent or SGD
     * @param data
     * @param l_features list of features
     * @param nfeatures
     * @param data_targets
     * @param l_targets
     * @param ntargets
     * @param alpha
     * @param lambda
     * @param ITER_MAX
     * @return
     */
    HashMap<String, double[]> optimize(ArrayList<double[]> data, List<Integer> l_features, int nfeatures,
                                       List<String> data_targets, List<String> l_targets, int ntargets, double alpha, double lambda, int ITER_MAX) {
        HashMap<String, double[]> mThetas = new HashMap<String, double[]>();
        for (int i = 0; i < ntargets; i++) {
            double[] thetas = new double[nfeatures+1];
            for (int j = 0; j <= nfeatures; j++) { //last theta is for the interncept
                // thetas[j] = Math.random();
                thetas[j] = 0.0;
            }
            mThetas.put(l_targets.get(i), thetas);
        }
        for (int i = 0; i < ITER_MAX; i++) {
            sgd_once(mThetas, data, l_features, nfeatures, data_targets,
                    l_targets, ntargets, alpha, lambda);
            Evaluate.computeClassificationError();
            // System.out.println("Iteration " + i + "...");
//			gd_batch(mThetas, data, l_features, nfeatures, targets, l_targets, ntargets, alpha, lambda);
            // printThetas(l_targets, ntargets, mThetas);
        }
        return mThetas;
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
//		new LR(args[0], 0.005, 0.000000, 5000);
//		new LR("/home/yandong/Dropbox/code/node/lr/wpbc.data_noid_wh", 0.005, 0.000000, 50);
        if (args.length < 1)
            new LogisticRegression("/Users/yandong/Dropbox/code/node/lr/wpbc_real_train.csv", "/Users/yandong/Dropbox/code/node/lr/wpbc_real_test.csv", 0.0005, 0.000001, 5000);
        else
            new LogisticRegression(args[0], args[1], 0.0005, 0.000001, 5000);
        long t2 = System.currentTimeMillis();
        System.out.println(t2 - t1);

        // for leaf data
        // batch: 100 0.055882353 25763 randomized
        // batch: 100 0.2 25763 nonrandomized
        // batch: 100 0.4 no-l2 no-random with normailization
        // batch: 200 0.08235294 5304 randomized
        // batch: 200 0.3 5304 nonrandomized
        // batch: 20000: 0.57
        // batch: 40000 0.655
        // sgd: 0.00005, 10000 0.15294118
        // sgd: 0.000005, 10000 0.34411764
        // sgd: 0.000001, 10000 0.5617647
        // sgd: 0.000000, 10000 0.7470588

        // sgd: 0.000000, 5000 0.747 w-normalization 8665
        // sgd: 0.000000, 10000 0.79 w-normalization 16133
        // sgd: 0.000000, 20000 0.826 w-normalization
        // sgd: 0.000000, 200k  0.911 w-normalization
        // sgd: 500k 0.929 w-norm
        // sgd: 20000: 0.7705882 34943 init:0.2
        // sgd: 20k 0.7705882 init:0.0
        // sgd: 20k 0.7705882 init:1.0
        // sgd: 40000: 0.817 69607
        // 60000 0.835 104159
        // 80000 0.84411764 139034
        // 200k 0.84411764 138863
    }
}

