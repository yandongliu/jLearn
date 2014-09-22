package learning.classifier.tree;

import learning.data.Data;
import learning.data.DefaultHashMap;
import learning.data.Pair;
import learning.io.DataReader;

import java.io.IOException;
import java.util.*;

/**
 * Created by yandong on 7/11/14.
 */
public class C45<T> {
    boolean DEBUG = false;

    void debug(String s) {
        if(DEBUG)
            System.out.println(s);
    }

    public C45(boolean debug) {
        this.DEBUG = debug;
    }

    Pair<ArrayList<double[]>, List<T>> cutDataLE(ArrayList<double[]> data, List<T> data_targets, int featureId, double cut) {
        ArrayList<double[]> set1 = new ArrayList<double[]>();
        List<T> target1 = new ArrayList<T>();
        for(int i=0;i<data.size();i++) {
            double[] row = data.get(i);
            T target = data_targets.get(i);
            if(row[featureId]<=cut) {
                set1.add(row);
                target1.add(target);
            }
        }
        return new Pair<ArrayList<double[]>, List<T>>(set1, target1);
    }

    Pair<ArrayList<double[]>, List<T>> cutDataGT(ArrayList<double[]> data, List<T> data_targets, int featureId, double cut) {
        ArrayList<double[]> set1 = new ArrayList<double[]>();
        List<T> target1 = new ArrayList<T>();
        for(int i=0;i<data.size();i++) {
            double[] row = data.get(i);
            T target = data_targets.get(i);
            if(row[featureId]>cut) {
                set1.add(row);
                target1.add(target);
            }
        }
        return new Pair<ArrayList<double[]>, List<T>>(set1, target1);
    }

    double condEntropy(ArrayList<double[]> data, List<T> data_targets, int featureId, double cut) {
        Pair<ArrayList<double[]>, List<T>> smaller = cutDataLE(data, data_targets, featureId, cut);
        Pair<ArrayList<double[]>, List<T>> bigger = cutDataGT(data, data_targets, featureId, cut);

        double n = data.size();
        return smaller.val1.size()/n*entropy(smaller.val2) + bigger.val1.size()/n*entropy(bigger.val2);
    }
    double entropy(List<T> data_targets) {
        HashMap<T, Integer> uniqVal = new HashMap<T, Integer>();
        for(int i=0;i<data_targets.size();i++) {
            T val = data_targets.get(i);
            if(!uniqVal.containsKey(val)) {
                uniqVal.put(val, 0);
            }
            uniqVal.put(val, uniqVal.get(val)+1);
        }
        double len = data_targets.size();
        double sum = 0.0;
        for(T key:uniqVal.keySet()) {
            double p = uniqVal.get(key)/len;
            sum += (-p * Math.log(p));
        }
        return sum;
    }

    ArrayList<Double> sampleFeatureValsByNum (HashSet<Double> vals, int cap) {
        ArrayList<Double> l = new ArrayList<Double>();
        double prob = (double)cap/vals.size();
        for(double d:vals) {
            if(Math.random()<prob) l.add(d);
        }

        return l;
    }


    ArrayList<Double> sampleFeatureValsByProb (HashSet<Double> vals, double prob) {
        ArrayList<Double> l = new ArrayList<Double>();
//        Random r = new Random(System.currentTimeMillis());
        if(vals.size()*prob <10.0) prob=1.0;
        for(double d:vals) {
            if(Math.random()<prob) l.add(d);
        }

        return l;
    }

    void sampleDataByProb (ArrayList<double[]> data, List<T> targets, double prob,ArrayList<double[]> d, List<T> t) {
        assert data.size()==targets.size();

        if(targets.size()*prob <10.0) prob=1.0;

        for(int i=0;i<targets.size();i++) {
            if(Math.random()<prob) {
                d.add(data.get(i));
                t.add(targets.get(i));
            }
        }
    }

    /**
     * compute gain for this feature
     * @return
     */
    Pair<Double, Double> gain(ArrayList<double[]> data, List<T> data_targets, int featureId, int numFeatureVal) {
        HashSet<Double> uniqVal = new HashSet<Double>();
        for(int i=0;i<data.size();i++) {
//            if(!uniqVal.contains(data.get(i)[featureId])) {
              uniqVal.add(data.get(i)[featureId]);
//            }
        }
        ArrayList<Double> listVal = sampleFeatureValsByNum(uniqVal, numFeatureVal);
//        System.out.println("#uniq feature val:"+listVal.size());
//        double
        double max = Integer.MIN_VALUE;
        double maxCut = 0;
        for(double cut:listVal) {
            double _gain =  - condEntropy(data, data_targets, featureId, cut);
            if(max<_gain) {
                max = _gain;
                maxCut = cut;
            }
        }
        return new Pair<Double, Double>(max,maxCut);
    }

    /**
     * Find the feature best gives max gain
     * @param data
     * @param data_targets
     * @return
     */
    Pair<Integer, Double> maxGain(ArrayList<double[]> data, List<T> data_targets, int nFeatures, int numFeatureVal, double featureSampleRate) {
//        double setEntropy = entropy(targets);
        int maxFeature = -1;
        double maxCut = 0;
        double maxGain = Integer.MIN_VALUE;
        for(int i=0;i<nFeatures;i++) {
            if(Math.random()<featureSampleRate) {
                Pair<Double, Double> gainPair = gain(data, data_targets, i, numFeatureVal);
//            System.out.println(gainPair.val1);
                if (maxGain < gainPair.val1) {
                    maxGain = gainPair.val1;
                    maxCut = gainPair.val2;
                    maxFeature = i;
                }
            }
        }
        debug("maxGain:"+maxFeature+"/"+ maxGain);
        return new Pair<Integer, Double>(maxFeature, maxCut);
    }

    public void printTree(TreeNode node, int depth) {
        if(node==null) return;
        for(int i=0;i<depth*2;i++) System.out.print(" ");
        if(node.type == TreeNode.NodeType.FEATURE)
            System.out.println(node.featureName+" "+node.cut);
        else
            System.out.println(node.featureName + " " + node.val);
        printTree(node.leftChild, depth+1);
        printTree(node.rightChild, depth+1);
    }

    void printRow(double[] data) {
        for (int i = 0; i < data.length; i++) {
            System.out.print(data[i] + "\t");
        }
    }

    void printData(ArrayList<double[]> data, List<String> targets) {
        int n1 = data.size();
        int n2 = targets.size();
        if(n1!=n2) {
            System.out.println("ERROR! Difft sizes!");
            System.exit(1);
        }
        for (int i = 0; i < n1; i++) {
            printRow(data.get(i));
            System.out.print("\t");
            System.out.println(targets.get(i));
        }
    }

    T getMajorClass(DefaultHashMap<T, Integer> mapTargets) {
        T majorClass = null;
        int majorCnt = 0;
        for(T a:mapTargets.keySet()) {
            if(majorCnt<mapTargets.get(a)) {
                majorCnt = mapTargets.get(a);
                majorClass = a;
            }
        }
        return majorClass;
    }

    public TreeNode findNode(ArrayList<double[]> _data, List<T> _data_targets, int nFeatures, List<String> l_featureNames, int numNodes, int numFeatureVal, double dataSampleRate, double featureSampleRate) {
        ArrayList<double[]> new_data = new ArrayList<double[]>();
        List<T> new_targets = new ArrayList<T>();
        sampleDataByProb(_data, _data_targets, dataSampleRate, new_data, new_targets);
        debug("data size:"+new_data.size());
//        HashSet<String> setTargets = new HashSet<String>();
        DefaultHashMap<T, Integer> mapTargets = new DefaultHashMap<T, Integer>(0);
        for(T a:new_targets) mapTargets.put(a, mapTargets.get(a)+1);
        debug("target uniq val:"+mapTargets.size());
        if(numNodes<=1) return new TreeNode("VAL",-1, getMajorClass(mapTargets));
        //if only 1 class exists in data
        if(mapTargets.size()==1) return new TreeNode("VAL",-1, new_targets.get(0));
//        if(data.size()<10) return null;
        Pair<Integer, Double> maxPair = maxGain(new_data, new_targets, nFeatures, numFeatureVal, featureSampleRate);//feature_id, cut
//        printData(data, targets);
        int f_id = maxPair.val1;
        double f_cut = maxPair.val2;
        String f_name = l_featureNames.get(f_id);
        TreeNode node = new TreeNode(f_name, f_id, f_cut);
        debug("new node:"+f_name+" / "+f_id+" "+f_cut);
        Pair<ArrayList<double[]>, List<T>> smaller = cutDataLE(new_data, new_targets, f_id, f_cut);
        Pair<ArrayList<double[]>, List<T>> bigger = cutDataGT(new_data, new_targets, f_id, f_cut);

        //data is not getting smaller
        if(smaller.val2.size()==0||bigger.val2.size()==0) {
            return new TreeNode("VAL",-1, getMajorClass(mapTargets));
        }
        node.setLeftChild(findNode(smaller.val1, smaller.val2, nFeatures, l_featureNames, numNodes/2, numFeatureVal, dataSampleRate, featureSampleRate));
        node.setRightChild(findNode(bigger.val1, bigger.val2, nFeatures, l_featureNames, numNodes/2, numFeatureVal, dataSampleRate, featureSampleRate));
        return node;
    }

    public<T> T predict(TreeNode root, double[] row) {
        TreeNode<T> node = root;
        while(node != null) {
            if(node.type == TreeNode.NodeType.FEATURE) {
                double data_val = row[node.featureId];
                if(data_val <= node.cut) {
                    node = node.leftChild;
                } else {
                    node = node.rightChild;
                }
            } else if (node.type == TreeNode.NodeType.TERMINAL) {
                return node.val;
            }
            else {
                System.err.println("Tree Model is illeagl.");
            }
        }
        return node.val;
    }

}
