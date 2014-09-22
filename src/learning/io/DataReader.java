package learning.io;

import learning.data.Data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by yandong on 7/11/14.
 */
public class DataReader {

    public static void main(String[] args) {
        try {
            Data d = DataReader.readFvFile("/Users/yandong/Dropbox/code/gbdt-master/data/sampledata/data.fv");
            System.out.println(d.nFeatures);
            System.out.println(d.nRows);
            for (int i = 0; i < d.nRows; i++) {
                System.out.println(d.targets.get(i));
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    public static Data<Double> readFvFile(String fn) throws IOException {
        Data<Double> data = new Data<Double>();
        BufferedReader br = new BufferedReader(new FileReader(fn));
        String[] aa = br.readLine().split("\t");
        int ncols = aa.length;
        ArrayList<Integer> f_colIndex = new ArrayList<Integer>();

        for (int i = 0; i < ncols; i++) {
            if (!aa[i].equals("label")) {
                f_colIndex.add(i);
                data.l_featureNames.add(aa[i]);
            }
            else
                data.labelCol = i;
        }
        if (data.labelCol == -1) {
            System.out.println("No label column!");
            br.close();
            System.exit(1);
        }

        data.nFeatures = f_colIndex.size();
        String s;

        // read content into data, targets
        while ((s = br.readLine()) != null) {
            aa = s.split("\t");
            for (int i = 0; i < aa.length; i++) aa[i] = aa[i].trim();
            double[] dd = new double[data.nFeatures];
            int cnt = 0;
            for (int i = 0; i < data.nFeatures; i++) {
                if (aa[f_colIndex.get(i)].equals("?")) dd[i] = 0;
                else try {
                    dd[i] = Double.parseDouble(aa[f_colIndex.get(i)]);
                } catch (NumberFormatException ex) {
//                    ex.printStackTrace();
                    System.out.println("format error");
                }
            }
            data.s_targets.add(Double.parseDouble(aa[data.labelCol]));
            data.data.add(dd);
            data.targets.add(Double.parseDouble(aa[data.labelCol]));
        }
        br.close();
        data.nTargets = data.s_targets.size();
        data.l_targets.addAll(data.s_targets);
        data.nRows = data.targets.size();
        return data;
    }

    public static Data<Double> readRealDataRealTargetFile(String fn) throws IOException {
        Data<Double> data = new Data<Double>();
        BufferedReader br = new BufferedReader(new FileReader(fn));
        String[] aa = br.readLine().split(",");
        int ncols = aa.length;
        ArrayList<Integer> f_colIndex = new ArrayList<Integer>();

        // features
//        int cnt=0;
        for (int i = 0; i < ncols; i++) {
            if (!aa[i].equals("label")) {
//                data.l_features.add(cnt);//feature is from 0 ... nFeatures -1
                f_colIndex.add(i);
                data.l_featureNames.add(aa[i]);
//                data.mapFeatureId2Name.put(cnt, aa[i]);
//                data.mapFeatureName2Id.put(aa[i], cnt);
//                cnt++;
            }
            else
                data.labelCol = i;
        }
        if (data.labelCol == -1) {
            System.out.println("No label column!");
            br.close();
            System.exit(1);
        }

        data.nFeatures = f_colIndex.size();
        String s;

        // read content into data, targets
        while ((s = br.readLine()) != null) {
            aa = s.split(",");
            for (int i = 0; i < aa.length; i++) aa[i] = aa[i].trim();
            //file is real data. this line is ignored
            if (aa[data.labelCol].equals("feature_type")) {
                continue;
            }
            double[] dd = new double[data.nFeatures];
            int cnt = 0;
            for (int i = 0; i < data.nFeatures; i++) {
                if (aa[f_colIndex.get(i)].equals("?")) dd[i] = 0;
                else
                    dd[i] = Double.parseDouble(aa[f_colIndex.get(i)]);
            }
            data.s_targets.add(Double.parseDouble(aa[data.labelCol]));
            data.data.add(dd);
            data.targets.add(Double.parseDouble(aa[data.labelCol]));
        }
        br.close();
        data.nTargets = data.s_targets.size();
        data.l_targets.addAll(data.s_targets);
        data.nRows = data.targets.size();
        return data;
    }
    /**
     * read in float feature file
     *
     * @param fn
     */
    public static Data<String> readRealDataFile(String fn) throws IOException {
        Data<String> data = new Data<String>();
        BufferedReader br = new BufferedReader(new FileReader(fn));
        String[] aa = br.readLine().split(",");
        int ncols = aa.length;
        ArrayList<Integer> f_colIndex = new ArrayList<Integer>();

        // features
//        int cnt=0;
        for (int i = 0; i < ncols; i++) {
            if (!aa[i].equals("label")) {
//                data.l_features.add(cnt);//feature is from 0 ... nFeatures -1
                f_colIndex.add(i);
                data.l_featureNames.add(aa[i]);
//                data.mapFeatureId2Name.put(cnt, aa[i]);
//                data.mapFeatureName2Id.put(aa[i], cnt);
//                cnt++;
            }
            else
                data.labelCol = i;
        }
        if (data.labelCol == -1) {
            System.out.println("No label column!");
            br.close();
            System.exit(1);
        }

        data.nFeatures = f_colIndex.size();
        String s;

        // read content into data, targets
        while ((s = br.readLine()) != null) {
            aa = s.split(",");
            for (int i = 0; i < aa.length; i++) aa[i] = aa[i].trim();
            //file is real data. this line is ignored
            if (aa[data.labelCol].equals("feature_type")) {
                continue;
            }
            double[] dd = new double[data.nFeatures];
            int cnt = 0;
            for (int i = 0; i < data.nFeatures; i++) {
                if (aa[f_colIndex.get(i)].equals("?")) dd[i] = 0;
                else
                  dd[i] = Double.parseDouble(aa[f_colIndex.get(i)]);
            }
            data.s_targets.add(aa[data.labelCol]);
            data.data.add(dd);
            data.targets.add(aa[data.labelCol]);
        }
        br.close();
        data.nTargets = data.s_targets.size();
        data.l_targets.addAll(data.s_targets);
        data.nRows = data.targets.size();
        return data;
    }
}
