package learning.regression.boosting;

import learning.classifier.tree.C45;
import learning.classifier.tree.TreeNode;
import learning.data.Data;
import learning.evaluate.Evaluate;
import learning.io.DataReader;

import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by yandong on 7/15/14.
 */
public class GradientBoosting {
    void initPrediction(double[] predictions) {
        //0.0
    }

    void sampleData(Data<Double> d) {

    }

    void reportProgress(int iter, Loss loss) {
        System.out.println(String.format("Iteration:%d MSE:%f time:100s", iter, loss.mse));
    }

    void gbdt(C45<Double> c45, Data<Double> d, ArrayList<TreeNode> trees, ArrayList<Double> weights, Loss loss) {
        double[] prediction = new double[d.nRows];
        double[] score = new double[d.nRows];
        double[] gradient = new double[d.nRows];
        double shrinkage = 1.0;
        initPrediction(prediction);

        sampleData(d);

        for(int i=0;i<loss.nTrees;i++) {
            System.out.println("Tree:"+i);
            loss.compGradient(d, prediction, gradient);
            ArrayList<Double> targets = new ArrayList<Double>();
//            for(int j=0;j<d.nRows;j++) targets.add(d.targets.get(j));

            for (int j = 0; j < d.nRows; j++) {
                score[j] = prediction[j];
            }

            TreeNode tree = c45.findNode(d.data, d.targets, d.nFeatures, d.l_featureNames, 10, 10, 0.3, 1.0);
            trees.add(tree);
            weights.add(0.05);

            reportProgress(i, loss);

            //update score
            
            //update prediction
            for (int j = 0; j < d.nRows; j++) {
                //prediction[j] += score[j]*shrinkage;
//                updatePrediction(c45, trees, weights, d, prediction);
                prediction[j] =+ score[j];
            }
        }
    }

    void updatePrediction(C45<Double> c45, ArrayList<TreeNode> trees, ArrayList<Double> weights, Data<Double> d, double[] prediction) {
        for (int i = 0; i < d.nRows; i++) {
            double finalResp = 0.0;
            for (int j = 0; j < trees.size(); j++) {
                double resp = c45.predict(trees.get(j), d.data.get(i));
                finalResp += resp*weights.get(j);
            }
            prediction[i] = finalResp;
        }
    }

    public void train(Data<Double> d) {
        Loss loss = new Loss();//regression loss
        ArrayList<TreeNode> trees = new ArrayList<TreeNode>();
        ArrayList<Double> weights = new ArrayList<Double>();
        C45 c45 = new C45(false);
//        gbdt(c45, d, trees, weights, loss);
        for (int i = 0; i < trees.size(); i++) {
//            c45.printTree(trees.get(i), 0);
        }


        double[] targets = new double[d.nRows];
        for(int i=0;i<d.nRows;i++) targets[i] = d.targets.get(i);
        double[] prediction = new double[d.nRows];
        Evaluate.computeRegressionError(targets, prediction);
        updatePrediction(c45, trees, weights, d, prediction);
        Evaluate.computeRegressionError(targets, prediction);
//        for(int i=0;i<d.nRows;i++) {
//            System.out.println(targets[i]+"\t"+prediction[i]);
//        }
        /*
        prediction == 0
        mae:1.4662162162162162
        rmse:1.6259093089963785
        t trees
        mae:0.8565033783783794
        rmse:0.9539967491020624
        9 trees
        mae:0.5439189189189191
        rmse:0.6523605784566731
        10 trees
        mae:0.47778716216216277
        rmse:0.6204768763250206
        11 trees
        mae:0.4792229729729736
        rmse:0.651751797393307
        15 trees
        mae:0.7908783783783793
        rmse:0.9594304335160293
        20 trees
        mae:1.5220439189189168
        rmse:1.6610476789941602
         */

    }

    public static void main(String[] args) {
        try {
            String fn_train="/Users/yandong/dev/ml/fv/with_grade.txt";
            Data<Double> data = DataReader.readFvFile(fn_train);
            GradientBoosting gb = new GradientBoosting();
            gb.train(data);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
}
