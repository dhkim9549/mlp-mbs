package com.dhkim9549.mlpmbs;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.*;
import java.io.*;

/**
 *  Building a Debt Repayment Prediction Model with MLP
 * @author Dong-Hyun Kim
 */
public class MLPMBS {

    static String hpId = "MLPMBS_h3_uSGD_mb16_ss16";

    //double learnigRate = Double.parseDouble(args[0]);
    static double learnigRate = 0.00025;

    // Number of sample size per iteration
    static long nSamples = 16;

    // Mini-batch size
    static int batchSize = 16;

    // Evaluation sample size
    static long nEvalSamples = 10000;

    // Number of input variables to the neural network
    static int numOfInputs = 3;

    // Number of output variables of the neural network
    static int numOfOutputs = 6;

    static LineNumberReader in = null;
    static BufferedWriter logOut = null;
    static String trainingDataInputFileName = "/down/mbs_data/beta_zero/training_data_20181120/TRAINING_DATAS_20181120_shuffled.txt";
    static String logFileName = "/down/mbs_data/beta_zero/training_data_20181120/TRAINING_DATAS_20181120_shuffled_training_log.txt";

    public static void main(String[] args) throws Exception {

        System.out.println("************************************************");
        System.out.println("hpId = " + hpId);
        System.out.println("Number of hidden layers = 3");
        System.out.println("learnigRate = " + learnigRate);
        System.out.println("Updater = " + "SGD");
        System.out.println("mini-batch size (batchSize) = " + batchSize);
        System.out.println("Number of sample size per iteration (nSamples) = " + nSamples);
        System.out.println("i >= 0");
        System.out.println("************************************************");

        MultiLayerNetwork model = getInitModel(learnigRate);
        //MultiLayerNetwork model = readModelFromFile("/down/sin/css_model_MLPMBS_h2_uSGD_mb16_ss16_200000.zip");

        NeuralNetConfiguration config = model.conf();
        System.out.println("config = " + config);

        // Training data input file reader
        in = new LineNumberReader(new FileReader(trainingDataInputFileName));
        logOut = new BufferedWriter(new FileWriter(logFileName));


        // Training iteration
        long i = 0;

        while(true) {

            i++;

            if(i % 100 == 0) {
                System.out.println("i = " + i);
                logOut.write("i = " + i + "\n");
                evaluateModel(model);
            }

            /*
            if(i % 1 == 0) {
                MLPMBSEval.evaluateModelBatch(model);
            }
            */

            List<DataSet> listDs = getTrainingData();
            DataSetIterator trainIter = new ListDataSetIterator(listDs, batchSize);

            // Train the model
            model = train(model, trainIter);

            if (i % 50000 == 0) {
                writeModelToFile(model, "/down/mbs_model_" + hpId + "_" + i + ".zip");
            }

            logOut.flush();
        }
    }

    public static MultiLayerNetwork getInitModel(double learningRate) throws Exception {

        int seed = 123;

        int numInputs = numOfInputs;
        int numOutputs = numOfOutputs;
        int numHiddenNodes = 30;

        System.out.println("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .updater(Updater.SGD)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        return model;
    }

    public static MultiLayerNetwork train(MultiLayerNetwork model, DataSetIterator trainIter) throws Exception {

        //model.setListeners(new ScoreIterationListener(1000));

        model.fit( trainIter );

        return model;
    }

    private static List<DataSet> getTrainingData() throws Exception {

        //System.out.println("Getting training data...");

        List<DataSet> listDs = new ArrayList<>();

        while(listDs.size() < nSamples) {

            String s = in.readLine();
            if(s == null) {
                //System.out.println("Training data file rollover...");
                in.close();
                in = new LineNumberReader(new FileReader(trainingDataInputFileName));
                continue;
            }
            if(s.indexOf("LOAN_AMT") >= 0) {
                continue;
            }

            DataSet ds = getDataSet(s);
            if(ds == null) {
            } else {
                listDs.add(ds);
            }
        }

        Collections.shuffle(listDs);

        //System.out.println("listDs.size() = " + listDs.size());
        //System.out.println("Getting training data complete.");

        return listDs;
    }

    public static String getToken(String s, int x) {
        return getToken(s, x, " \t\n\r\f");
    }

    public static String getToken(String s, int x, String delim) {

        s = s.replaceAll("\t", "\t ");

        StringTokenizer st = new StringTokenizer(s, delim);
        int counter = 0;
        String answer = null;
        while(st.hasMoreTokens()) {
            String token = st.nextToken();
            if(counter == x) {
                answer = token.trim();
            }
            counter++;
        }
        return answer;
    }

    private static DataSet getDataSet(String s) throws Exception {

        String loan_acc_no = getToken(s, 19, "\t");
        String treat_dy = getToken(s, 6, "\t");

        logOut.write("loan_acc_no = " + loan_acc_no + "\n");
        logOut.write("treat_dy = " + treat_dy + "\n");



        double[] featureData = new double[numOfInputs];

        double loan_amt = Double.parseDouble(getToken(s, 0, "\t"));
        featureData[0] = rescaleAmt(loan_amt, 0, 500000000);

        double loan_mms_cnt = Double.parseDouble(getToken(s, 5, "\t"));
        featureData[1] = rescaleAmt(loan_mms_cnt, 0, 480);

        double loan_rat = Double.parseDouble(getToken(s, 1, "\t"));
        featureData[2] = rescaleAmt(loan_rat, 0, 20);






        /*
        double cllct_rate = Double.parseDouble(getToken(s, 18, "\t"));
        double cllct_rate_old = Double.parseDouble(getToken(s, 17, "\t"));
        long debt_ramt = Long.parseLong(getToken(s, 16, "\t"));
        long dischrg_dur_month = Long.parseLong(getToken(s, 3, "\t"));
        long org_guarnt_dur_month = Long.parseLong(getToken(s, 2, "\t"));
        String guarnt_dvcd_rent_yn = getToken(s, 4, "\t");
        String guarnt_dvcd_mid_yn = getToken(s, 5, "\t");
        String guarnt_dvcd_buy_yn = getToken(s, 6, "\t");
        String crdrc_yn = getToken(s, 7, "\t");
        String revivl_yn = getToken(s, 8, "\t");
        String exempt_yn = getToken(s, 9, "\t");
        String sptrepay_yn = getToken(s, 10, "\t");
        String psvact_yn = getToken(s, 11, "\t");
        long rdbtr_1_cnt = Long.parseLong(getToken(s, 12, "\t")); // new input
        long rdbtr_2_cnt = Long.parseLong(getToken(s, 13, "\t")); // new input
        long age = Long.parseLong(getToken(s, 14, "\t")); // new input
        long dischrg_occr_amt = Long.parseLong(getToken(s, 15, "\t")); // new input
        String prscp_cmplt_yn = getToken(s, 19, "\t"); // new input
        String ibon_amtz_yn = getToken(s, 20, "\t"); // new input
        long rdbtr_3_cnt = Long.parseLong(getToken(s, 21, "\t")); // new input
        */


        /*
        featureData[0] = cllct_rate_old;
        featureData[1] = rescaleAmt(debt_ramt);
        featureData[3] = rescaleAmt(org_guarnt_dur_month, 0, 120, true);
        featureData[4] = rescaleYn(guarnt_dvcd_rent_yn);
        featureData[5] = rescaleYn(guarnt_dvcd_mid_yn);
        featureData[6] = rescaleYn(guarnt_dvcd_buy_yn);
        featureData[7] = rescaleYn(crdrc_yn);
        featureData[8] = rescaleYn(revivl_yn);
        featureData[9] = rescaleYn(exempt_yn);
        featureData[10] = rescaleYn(sptrepay_yn);
        featureData[11] = rescaleYn(psvact_yn);
        featureData[12] = rescaleNum(rdbtr_1_cnt); // new input
        featureData[13] = rescaleNum(rdbtr_2_cnt); // new input
        featureData[14] = rescaleAmt(age, 0, 100); // new input
        featureData[15] = rescaleAmt(dischrg_occr_amt); // new input
        featureData[16] = rescaleYn(prscp_cmplt_yn); // new input
        featureData[17] = rescaleYn(ibon_amtz_yn); // new input
        featureData[18] = rescaleNum(rdbtr_3_cnt); // new input
        */



        String[] loan_ramt_str = new String[14];
        double[] loan_ramt = new double[14];
        for(int i = 0; i < loan_ramt_str.length; i++) {
            loan_ramt_str[i] = getToken(s, 20 + i, "\t");
            if(loan_ramt_str[i].equals("")) {
                loan_ramt[i] = 0;
            } else {
                loan_ramt[i] = Long.parseLong(loan_ramt_str[i]);
            }
        }

        int treat_year = (int)Long.parseLong(treat_dy.substring(0, 4));

        double[] labelData = new double[numOfOutputs];
        double labelDataSum = 0.0;
        for(int i = 0; i < labelData.length; i++) {
            /*
            System.out.println("treat_year = " + treat_year);
            System.out.println("i = " + i);
            System.out.println("index = " + (i + (treat_year - 2004) + 1));
            */
            labelData[i] = rescaleAmt(loan_ramt[i + (treat_year - 2004) + 1], 0, 500000000);
            labelDataSum += labelData[i];
        }

        INDArray feature = Nd4j.create(featureData, new int[]{1, numOfInputs});
        INDArray label = Nd4j.create(labelData, new int[]{1, numOfOutputs});

        DataSet ds = null;
        if(loan_rat <= 0.0) {
        } else {
            ds = new DataSet(feature, label);
        }

        /*
        System.out.println("\n guarnt_no = " + guarnt_no);
        System.out.println(rdbtr_2_cnt + " " + age + " " + dischrg_occr_amt + " " + prscp_cmplt_yn + " " + ibon_amtz_yn);
        System.out.println(cllct_rate);
        System.out.println("ds = " + ds);
        */

        // System.out.println("\n ds ");
        // System.out.println( ds);
        logOut.write("\n ds \n");
        logOut.write(ds + "\n");


        return ds;
    }

    public static void evaluateModel(MultiLayerNetwork model) throws Exception {

        System.out.println(">>> Evaluating...");

        double[] featureData = new double[numOfInputs];
        featureData[0] = 0.46;
        featureData[1] = 0.75;
        featureData[2] = 0.75;





        INDArray feature = Nd4j.create(featureData, new int[]{1, numOfInputs});
        INDArray output = model.output(feature);
        //System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> feature = " + feature);
        System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> output = " + output);
        logOut.write(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> output = " + output);
    }

    public static double rescaleAmt(long x) {
        return rescaleAmt(x, 0, 100000000);
    }

    public static double rescaleAmt(double x, double min, double max) {
        return rescaleAmt(x, min, max, false);
    }

    public static double rescaleAmt(double x, double min, double max, boolean forceMin) {
        if(forceMin) {
            if(x < min) {
                x = min;
            }
        }
        double base = (max - min) / 10.0;
        double y = (Math.log(x - min + base) - Math.log(base)) / (Math.log(max - min + base) - Math.log(base));
        return y;
    }

    public static double rescaleYn(String x) {
        double y = 0.0;
        if(x.equals("Y")) {
            y = 1.0;
        }
        return y;
    }

    public static double rescaleNum(long x) {
        double y = 0.0;
        if(x > 0) {
            y = 1.0;
        }
        return y;
    }

    public static MultiLayerNetwork readModelFromFile(String fileName) throws Exception {

        System.out.println("Deserializing model...");

        // Load the model
        File locationToSave = new File(fileName);
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

        System.out.println("Deserializing model complete.");

        return model;
    }

    public static void writeModelToFile(MultiLayerNetwork model, String fileName) throws Exception {

        System.out.println("Serializing model...");

        // Save the model
        File locationToSave = new File(fileName); // Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true; // Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(model, locationToSave, saveUpdater);

        System.out.println("Serializing model complete.");

    }
}