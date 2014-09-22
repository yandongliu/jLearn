package learning.io;

import learning.data.Data;
import learning.util.StringUtils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by yandong on 7/15/14.
 */
public class WekaConverter {
    public void convertToWeka(String fn_fv, String fn_weka) {
        try {
            Data<Double> m = DataReader.readFvFile(fn_fv);
            BufferedWriter writer = new BufferedWriter(new FileWriter(fn_weka));
            writer.write("@relation train");
            writer.newLine();
            writer.newLine();
            for (int i = 0; i < m.nFeatures; i++) {
                writer.write("@attribute "+m.l_featureNames.get(i)+ " real");
                writer.newLine();
            }
            writer.newLine();
            ArrayList<Double> targets = new ArrayList<Double>(m.s_targets);
            writer.write("@attribute class {"+StringUtils.strJoin(targets,",")+"}");
            writer.newLine();
            writer.write("@data");
            writer.newLine();
            for (int i = 0; i < m.nRows; i++) {
                ArrayList<Double> al = new ArrayList<Double>();
                double[] row = m.data.get(i);
                for (int j = 0; j < m.nFeatures; j++) {
                    al.add(row[j]);
                }
                al.add(m.targets.get(i));
//                System.out.println(StringUtils.strJoin(al, ","));
                writer.write(StringUtils.strJoin(al,","));
                writer.newLine();
            }
            writer.close();
        } catch (IOException ex) {
            ex.printStackTrace();;
        }
    }
    public static void main(String[] args) {
        String fn_fv="/Users/yandong/dev/ml/fv/with_grade.txt";
        String fn_weka="/Users/yandong/dev/ml/fv/with_grade.arff";
        new WekaConverter().convertToWeka(fn_fv, fn_weka);
        System.out.println("finished converting");
    }
}
