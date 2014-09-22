package learning.classifier.tree;

import learning.data.Data;
import learning.evaluate.Evaluate;
import learning.io.DataReader;

import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by yandong on 7/16/14.
 */
public class TestTree {
    public void testFV(String fn_train, String fn_test) {
        C45<Double> tree = new C45<Double>(false);
        try {
            Data<Double> data = DataReader.readFvFile(fn_train);
            System.out.println("# rows:"+ data.nRows);
            System.out.println("# features:"+ data.nFeatures);
            for(String key: data.l_featureNames) {
//                System.out.println(key);
            }
            TreeNode<Double> root = tree.findNode(data.data, data.targets, data.nFeatures, data.l_featureNames, 200, 100, 0.6, 1.0);
            tree.printTree(root, 0);
            Data<Double> testData = DataReader.readFvFile(fn_test);
            int correct = 0;
            ArrayList<Double> alPred = new ArrayList<Double>();
            for(int i=0;i< testData.data.size();i++) {
                Double pred = tree.predict(root, testData.data.get(i));
                alPred.add(pred);
                if(pred.equals(testData.targets.get(i))) {
                    correct++;
                }
//                System.out.println("prediction:"+predict(root, testData.data.get(i))+" actual:"+testData.targets.get(i));
            }
            Evaluate.computeError(testData.targets, alPred);
            System.out.println("#examples:"+ testData.nRows+" accuracy:"+(double)correct/ testData.nRows);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    public void testCSV(String fn_train, String fn_test) {
        C45<String> tree = new C45<String>(false);

        try {
            Data<String> data = DataReader.readRealDataFile(fn_train);
            for(String key: data.l_featureNames) {
//                System.out.println(key);
            }
            TreeNode<Double> root = tree.findNode(data.data, data.targets, data.nFeatures, data.l_featureNames, 100, 20, 1, 1.0);
            tree.printTree(root, 0);
            Data<String> testData = DataReader.readRealDataFile(fn_test);
            int correct = 0;
            for(int i=0;i< testData.data.size();i++) {
                String pred = tree.predict(root, testData.data.get(i));
                if(pred.equals(testData.targets.get(i))) {
                    correct++;
                }
//                System.out.println("prediction:"+predict(root, testData.data.get(i))+" actual:"+testData.targets.get(i));
            }
            System.out.println("#examples:"+ testData.nRows+" accuracy:"+(double)correct/ testData.nRows);
        } catch (IOException ex) {
            ex.printStackTrace();
        }


    }
    public static void main(String[] args) {
        String fn1="/Users/Yandong/tmp/wpbc_real_train.csv";
        String fn2="/Users/Yandong/tmp/wpbc_real_test.csv";
//        new C45<Double>().testCSV(fn1, fn2);
//        new C45().testFV("/Users/yandong/Dropbox/code/gbdt-master/data/sampledata/data.fv", "/Users/yandong/Dropbox/code/gbdt-master/data/sampledata/data.fv");
        new TestTree().testFV("/Users/yandong/dev/ml/fv/with_grade.txt", "/Users/yandong/dev/ml/fv/with_grade.txt");
    }
}
