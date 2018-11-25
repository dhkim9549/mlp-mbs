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

            String loan_acc_no = MLPMBS.getToken(s, 19, "\t");

            double[] featureData = new double[MLPMBS.numOfInputs];

            double loan_amt = Double.parseDouble(MLPMBS.getToken(s, 0, "\t"));
            featureData[0] = MLPMBS.rescaleAmt(loan_amt, 0, 500000000);

            double loan_mms_cnt = Double.parseDouble(MLPMBS.getToken(s, 5, "\t"));
            featureData[1] = MLPMBS.rescaleAmt(loan_mms_cnt, 0, 480);

            double loan_rat = Double.parseDouble(MLPMBS.getToken(s, 1, "\t"));
            featureData[2] = MLPMBS.rescaleAmt(loan_rat, 0, 20);

            double hold_mms_cnt = Double.parseDouble(MLPMBS.getToken(s, 8, "\t"));
            featureData[3] = MLPMBS.rescaleAmt(hold_mms_cnt, 0, 60);

            double edappnt_repay_amt = Double.parseDouble(MLPMBS.getToken(s, 9, "\t"));
            featureData[4] = MLPMBS.rescaleAmt(edappnt_repay_amt, 0, 500000000);




            INDArray feature = Nd4j.create(featureData, new int[]{1, MLPMBS.numOfInputs});
            INDArray output = model.output(feature);


            String s2 = "";
            s2 += loan_acc_no + "\t";
            s2 += loan_amt + "\t";
            s2 += loan_mms_cnt + "\t";
            s2 += loan_rat + "\t";

            s2 += output.getDouble(0) + "\t";
            s2 += loan_amt * output.getDouble(0) + "\t";

            out.write(s2 + "\n");
            out.flush();

            i++;
        }

        out.close();
    }
}