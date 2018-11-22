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
    static String testDataInputFileName = "/down/mbs_data/beta_zero/training_data_20181120/TRAINING_DATAS_20181120.txt";
    static String modelInputFileName = "/down/model/drp_model_MLPDRP_h3_uSGD_mb16_ss16_3250000.zip";

    public static void main(String[] args) throws Exception {

        if(args.length >= 1) {
            modelInputFileName = args[0];
        }

        MultiLayerNetwork model = MLPMBS.readModelFromFile(modelInputFileName);

        evaluateModelBatch(model);

    }

    public static void evaluateModelBatch(MultiLayerNetwork model) throws Exception {

        System.out.println("Evaluating batch...");

        // Training data input file reader
        in = new LineNumberReader(new FileReader(testDataInputFileName));

        // Evaluation result output writer
        BufferedWriter out = new BufferedWriter(new FileWriter("/down/mbs_data/beta_zero/training_data_20181120/test_result.txt"));

        String header = "";
        header += "seq\t";
        header += "loan_acct\t";
        header += "loan_ramt_rat_1\t";
        header += "loan_ramt_rat_2\t";
        header += "loan_ramt_rat_3\t";
        header += "loan_ramt_rat_4\t";
        header += "loan_ramt_rat_5\t";
        header += "loan_ramt_rat_6\t";
        out.write(header + "\n");
        out.flush();

        int i = 0;
        Random rand = new Random();

        String s = "";
        while((s = in.readLine()) != null) {

            if(s.indexOf("LOAN_AMT") >= 0) {
                continue;
            }

            long seq = rand.nextLong();
            String loan_acc_no = MLPMBS.getToken(s, 19, "\t");

            double[] featureData = new double[MLPMBS.numOfInputs];

            double loan_amt = Double.parseDouble(MLPMBS.getToken(s, 0, "\t"));
            featureData[0] = MLPMBS.rescaleAmt(loan_amt, 0, 500000000);

            double loan_mms_cnt = Double.parseDouble(MLPMBS.getToken(s, 5, "\t"));
            featureData[1] = MLPMBS.rescaleAmt(loan_mms_cnt, 0, 480);





            INDArray feature = Nd4j.create(featureData, new int[]{1, MLPMBS.numOfInputs});
            INDArray output = model.output(feature);



            if(i % 50000 == 0) {
                System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> test i = " + i);
                System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> feature = " + feature);
                System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> output = " + output);
            }

            String s2 = "";
            s2 += seq + "\t";
            s2 += loan_acc_no + "\t";

            for(int j = 0; j < output.size(0); j++) {
                s2 += output.getDouble(j) + "\t";
            }

            out.write(s2 + "\n");
            out.flush();

            i++;
            break;
        }

        out.close();
    }
}