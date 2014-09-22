package learning.data;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

/**
 * Created by yandong on 7/11/14.
 */
public class Data<T> {
    public ArrayList<double[]> data; //double data
//    public List<Integer> l_features; //list of features ids
    public List<String> l_featureNames; //list of features ids
    public HashSet<T> s_targets; //set of targets
    public List<T> l_targets; //list of targets
    public List<T> targets; //target data. doing classification here
//    public HashMap<Integer, String> mapFeatureId2Name; //map of feature's id to its name. id is assigned internally
//    public HashMap<String, Integer> mapFeatureName2Id; //map of feature's id to its name. id is assigned internally
    public int labelCol; //column for feature
    public int nFeatures;
    public int nTargets;
    public int nRows;
    public Data() {
        data = new ArrayList<double[]>();
//        l_features = new ArrayList<Integer>();
        l_featureNames = new ArrayList<String>();
        targets = new ArrayList<T>();
        l_targets = new ArrayList<T>();
        s_targets = new HashSet<T>();
//        mapFeatureId2Name =  new HashMap<Integer, String>();
//        mapFeatureName2Id = new HashMap<String, Integer>();
        labelCol = -1;
    }
}
