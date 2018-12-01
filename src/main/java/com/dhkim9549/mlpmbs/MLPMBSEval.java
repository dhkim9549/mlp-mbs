package com.dhkim9549.mlpmbs;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import scala.util.Random;

import java.io.*;

/**
 * Evaluate a trained MLP MBS model
 */
public class MLPMBSEval {

    static LineNumberReader in = null;
    static String testDataInputFileName = "/down/mbs_data/beta_zero/testing_data_20181122/TESTING_DATA_20181122.txt";
//    static String testDataInputFileName = "/down/mbs_data/beta_zero/training_data_20181120/TRAINING_DATAS_20181120_shuffled_2.txt";
    static String modelInputFileName = "/down/model/drp_model_MLPDRP_h3_uSGD_mb16_ss16_3250000.zip";

    public static void main(String[] args) throws Exception {

        if(args.length >= 1) {
            modelInputFileName = args[0];
        }

        MultiLayerNetwork model = MLPMBS.readModelFromFile(modelInputFileName);

        evaluateModelBatch(model);

    }

    public static void evaluateModelBatch(MultiLayerNetwork model) throws Exception {

        System.out.println("Evaluating with test data...");

        // Training data input file reader
        in = new LineNumberReader(new FileReader(testDataInputFileName));

        // Evaluation result output writer
        BufferedWriter out = new BufferedWriter(new FileWriter("/down/mbs_data/beta_zero/testing_data_20181122/TESTING_DATA_20181122_test_result_"
                + "numOfInputs_" + MLPMBS.numOfInputs +"_iteration_" + MLPMBS.cnt + ".txt"));

        String header = "";
        header += "loan_acct\t";
        header += "loan_amt\t";
        header += "loan_mms_cnt\t";
        header += "loan_rat\t";
        header += "hold_mms_cnt\t";
        header += "edappnt_repay_amt\t";
        header += "repay_mthd_cd\t";

        header += "loan_ramt_rat\t";
        header += "loan_ramt\t";
        out.write(header + "\n");
        out.flush();

        int i = 0;

        String s = "";
        while((s = in.readLine()) != null) {

            if(s.indexOf("LOAN_AMT") >= 0) {
                continue;
            }

            if(i % 1000 == 0) {
                System.out.println("i = " + i);
            }

            String loan_acc_no = MLPMBS.getToken(s, 19, "\t");

            double[] featureData = new double[MLPMBS.numOfInputs];

            double loan_amt = Double.parseDouble(MLPMBS.getToken(s, 0, "\t"));
            featureData[0] = MLPMBS.rescaleAmt(loan_amt, 0, 500000000);

            double loan_mms_cnt = Double.parseDouble(MLPMBS.getToken(s, 5, "\t"));
            featureData[1] = MLPMBS.rescaleAmt(loan_mms_cnt, 0, 480);

            double loan_rat = Double.parseDouble(MLPMBS.getToken(s, 1, "\t"));
            featureData[2] = MLPMBS.rescaleAmt(loan_rat, 0, 100);

            double hold_mms_cnt = Double.parseDouble(MLPMBS.getToken(s, 8, "\t"));
            featureData[3] = MLPMBS.rescaleAmt(hold_mms_cnt, 0, 60);

            String edappnt_repay_amt_str = MLPMBS.getToken(s, 9, "\t");
            double edappnt_repay_amt = 0.0;
            if(edappnt_repay_amt_str.equals("")) {
                edappnt_repay_amt = 0.0;
            } else {
                edappnt_repay_amt = Double.parseDouble(MLPMBS.getToken(s, 9, "\t"));
            }
            featureData[4] = MLPMBS.rescaleAmt(edappnt_repay_amt, 0, 500000000);

            String repay_mthd_cd = MLPMBS.getToken(s, 7, "\t");
            if(repay_mthd_cd.equals("BL")) {
                featureData[5] = 1.0;
            } else {
                featureData[5] = 0.0;
            }
            if(repay_mthd_cd.equals("IB")) {
                featureData[6] = 1.0;
            } else {
                featureData[6] = 0.0;
            }
            if(repay_mthd_cd.equals("OB")) {
                featureData[7] = 1.0;
            } else {
                featureData[7] = 0.0;
            }
            if(repay_mthd_cd.equals("PI")) {
                featureData[8] = 1.0;
            } else {
                featureData[8] = 0.0;
            }
            if(repay_mthd_cd.equals("PL")) {
                featureData[9] = 1.0;
            } else {
                featureData[9] = 0.0;
            }
            if(repay_mthd_cd.equals("PO")) {
                featureData[10] = 1.0;
            } else {
                featureData[10] = 0.0;
            }
            if(repay_mthd_cd.equals("SU")) {
                featureData[11] = 1.0;
            } else {
                featureData[11] = 0.0;
            }



            INDArray feature = Nd4j.create(featureData, new int[]{1, MLPMBS.numOfInputs});
            INDArray output = model.output(feature);


            String s2 = "";
            s2 += loan_acc_no + "\t";
            s2 += loan_amt + "\t";
            s2 += loan_mms_cnt + "\t";
            s2 += loan_rat + "\t";
            s2 += hold_mms_cnt + "\t";
            s2 += edappnt_repay_amt + "\t";
            s2 += repay_mthd_cd + "\t";

            s2 += output.getDouble(0) + "\t";
            s2 += loan_amt * output.getDouble(0) + "\t";

            out.write(s2 + "\n");
            out.flush();

            i++;
        }

        out.close();
    }
}