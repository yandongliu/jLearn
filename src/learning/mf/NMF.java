package learning.mf;

import java.io.*;
import java.util.*;

class Rating {
    int uid, iid,rating;
    public Rating(int uid, int iid, int r) {
        this.uid = uid;
        this.iid = iid;
        this.rating = r;
    }
}
class Matrix {
    //int length_p, length_q;
    //ArrayList<Integer> rows;
    ArrayList<Rating> ratings;
    int rank;
    double[][] P, Q;
    int numUsers, numItems;
    public Matrix(int rank) {
        ratings = new ArrayList<Rating>();
        this.rank = rank;
    }
}

public class NMF {

    Matrix M;
//    ArrayList<double[]> P, Q;
    int numUsers, numItems;
    HashMap<String, Integer> mapUser2UID, mapItem2IID;
    boolean DEBUG = false;

    public NMF(int k) {
        M = new Matrix(k);
//        P = new ArrayList<double[]>();
//        Q = new ArrayList<double[]>();
        numUsers = 0;
        numItems = 0;
        mapUser2UID = new HashMap<String, Integer>();
        mapItem2IID = new HashMap<String, Integer>();
    }

    public void loadMatrixFromMovielens(String fn) {

        try {
            BufferedReader br = new BufferedReader(new FileReader(fn));
            String s;
            //6040 x 3706
            while((s = br.readLine())!=null) {
                String[] aa = s.split("::");
                String user = aa[0];
                String item = aa[1];
                int r = Integer.parseInt(aa[2]);
//                System.out.println(uid+" "+iid+" "+r);
//                if(numUsers < uid) numUsers = uid;
//                if(numItems < iid) numItems = iid;
                if(!mapUser2UID.containsKey(user)) {
                    mapUser2UID.put(user, numUsers++);
                }
                if(!mapItem2IID.containsKey(item)) {
                    mapItem2IID.put(item, numItems++);
                }
                int uid = mapUser2UID.get(user);
                int iid = mapItem2IID.get(item);
                M.ratings.add(new Rating(uid,iid,r));
            }
            M.numItems = numItems;
            M.numUsers = M.numUsers;
            M.P = new double[numUsers][M.rank];
            M.Q = new double[numItems][M.rank];
			Random r = new Random();
            for(int i=0;i<numUsers;i++) {
                for (int j = 0; j < M.rank; j++) {
                    M.P[i][j] = r.nextDouble();
                }
            }
            for(int i=0;i<numItems;i++) {
                for (int j = 0; j < M.rank; j++) {
                    M.Q[i][j] = r.nextDouble();
                }
            }
            System.out.println("#users:"+numUsers+" #items:"+numItems);
            br.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    double innerproduct (double[] a, double[] b) throws Exception {
        int n = a.length;
        if(n!=b.length) throw new Exception("a and b are not of same dimension.");
        double prdt = 0.0;
        for (int i = 0; i < n; i++) {
            prdt += a[i]*b[i];
        }
        return prdt;
    }

    void printArray(double[] a) {
        for (int i = 0; i < a.length; i++) {
            System.out.print(a[i]+" ");
        }
        System.out.println();
    }

    public void sgd(int num_iters)  {
        //M = P x Q
        //for each rating r in M, update P(user),Q(item)
        //q[item] = q[item] + \gamma * (p[user]*error - \lambda q[item])
        //p[user] = p[user] + \gamma * (q[item]*error - \lambda p[user])
        try {
            double minrmse = Integer.MAX_VALUE;
            double gamma = 0.00005;
            double lambda = 0.0000001;
            for (int iter=0;iter<num_iters;iter++) {
                double rmse = 0.0;
                for (int i = 0; i < M.ratings.size(); i++) {
                    Rating r = M.ratings.get(i);
                    int uid = r.uid;
                    int iid = r.iid;
                    int rating = r.rating;
                    double[] p = M.P[uid];
                    double[] q = M.Q[iid];
                    double predict = innerproduct(p, q);
                    double err = rating - predict;
//                System.out.println("predict/error:"+predict+"/"+err);
//                System.out.println("error:"+err);
                    rmse += err * err;
                    for (int j = 0; j < M.rank; j++) {
                        q[j] += (gamma * (p[j] * err - lambda * q[j]));
                        p[j] += (gamma * (q[j] * err - lambda * p[j]));
                    }
//                printArray(p);
//                printArray(q);
                    if (DEBUG&&i % 1000 == 0) {
                        System.out.println("processed " + i + " ratings.");
                    }
                }
                if (DEBUG) {
                    for (int i = 0; i < M.numUsers; i++) {
                        printArray(M.P[i]);
                    }
                    for (int i = 0; i < M.numItems; i++) {
                        printArray(M.Q[i]);
                    }
                }
                if(minrmse>rmse) minrmse = rmse;
                System.out.println(" rmse:" + rmse);
//                if(iter%50==0) System.out.println();
            }
            System.out.println();
            System.out.println("minrmse:"+minrmse/M.ratings.size());
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public void loadMatrixFromFile(String fn) {
    }

    public void test(int iters) {
        this.sgd(iters);
    }

    public static void main(String[] args) {
        NMF nmf = new NMF(20);
        if (args.length < 1)
            nmf.loadMatrixFromMovielens("/Users/yandongliu/work/2019_aug_jam/ml-1m/ratings.dat");
        else
            nmf.loadMatrixFromMovielens(args[0]);
        System.out.println("loaded");
        nmf.test(50);
    }
}
