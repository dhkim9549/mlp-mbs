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
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import java.util.*;
import java.io.*;

/**
 *  Building a Debt Repayment Prediction Model with MLP
 * @author Dong-Hyun Kim
 */
public class MLPMBS {

    static String hpId = "MLP_MBS_h3_uSGD_mb16_ss16";

    // Number of sample size per iteration
    static long nSamples = 16;

    // Mini-batch size
    static int batchSize = 16;

    // Evaluation sample size
    static long nEvalSamples = 10000;

    // Number of input variables to the neural network
    static int numOfInputs = 1;

    // Number of output variables of the neural network
    static int numOfOutputs = 2;

    // Number of hidden nodes at each layer
    static int numOfHiddenNodes = 30;

    static LineNumberReader in = null;
    static BufferedWriter logOut = null;
    static String trainingDataInputFileName = "/down/mbs_data/beta_zero/training_data_20181120/TRAINING_DATAS_20181120_shuffled.txt";
    static String logFileName = "/down/mbs_data/beta_zero/training_data_20181120/TRAINING_DATAS_20181120_shuffled_training_log.txt";

    // Training iteration
    static long cnt = 0;

    static double sum_loan_amt = 0;
    static double sum_ramt = 0;


    public static void main(String[] args) throws Exception {

        System.out.println("************************************************");
        System.out.println("hpId = " + hpId);
        System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Number of hidden layers = 3");
        System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Number of hidden nodes = " + numOfHiddenNodes);
        System.out.println("mini-batch size (batchSize) = " + batchSize);
        System.out.println("Number of sample size per iteration (nSamples) = " + nSamples);
        System.out.println("i >= 0");
        System.out.println("************************************************");

        MultiLayerNetwork model = getInitModel();
        //MultiLayerNetwork model = readModelFromFile("/down/sin/css_model_MLPMBS_h2_uSGD_mb16_ss16_200000.zip");

        NeuralNetConfiguration config = model.conf();
        System.out.println("config = " + config);

        // Training data input file reader
        in = new LineNumberReader(new FileReader(trainingDataInputFileName));
        logOut = new BufferedWriter(new FileWriter(logFileName));


        while(true) {

            if(cnt % 100 == 0) {
                System.out.println("cnt = " + cnt);
                System.out.println("sum_ramt = " + sum_ramt);
                System.out.println("sum_loan_amt = " + sum_loan_amt);
                System.out.println("(sum_ramt / sum_loan_amt) = " + ((double)sum_ramt / (double)sum_loan_amt));
            }
            if(cnt % 1 == 0) {
                logOut.write("cnt = " + cnt + "\n");
                evaluateModel(model);
            }
            if(cnt % 100 == 0) {
                System.out.println(new Date());
                logOut.write(new Date() + "\n");
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

            if (cnt % 50000 == 0) {
                writeModelToFile(model, "/down/mbs_model_" + hpId + "_" + cnt + ".zip");
            }

            logOut.flush();

            cnt++;
        }
    }

    public static MultiLayerNetwork getInitModel() throws Exception {

        int seed = 123;

        int numInputs = numOfInputs;
        int numOutputs = numOfOutputs;
        int numHiddenNodes = numOfHiddenNodes;

        Map<Integer, Double> lrSchedule = new HashMap<>();
        for(int i = 0; i <= 10; i++) {
            double learningRate = 0.0025 * ((10.0 - (double)i) / 10.0) + 0.000025 * (((double)i) / 10.0);
            lrSchedule.put(i * 1000, learningRate); // iteration #, learning rate
        }
        System.out.println("lrSchedule = " + lrSchedule);

        System.out.println("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
//                .l1(0.02)
                .updater(new Nesterovs(new MapSchedule(ScheduleType.ITERATION, lrSchedule)))
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
                .build();

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
                System.out.println("Training data file rollover...");
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

        /*
        double loan_mms_cnt = Double.parseDouble(getToken(s, 5, "\t"));
        featureData[1] = rescaleAmt(loan_mms_cnt, 0, 480);

        double loan_rat = Double.parseDouble(getToken(s, 1, "\t"));
        featureData[2] = rescaleAmt(loan_rat, 0, 20);
        */

        String loan_ramt_str_sum = "";
        String[] loan_ramt_str = new String[14];
        double[] loan_ramt = new double[14];
        for(int i = 0; i < loan_ramt_str.length; i++) {
            loan_ramt_str[i] = getToken(s, 20 + i, "\t");
            loan_ramt_str_sum += loan_ramt_str[i];
            if(loan_ramt_str[i].equals("")) {
                loan_ramt[i] = 0;
            } else {
                loan_ramt[i] = Long.parseLong(loan_ramt_str[i]);
            }
        }

        int treat_year = (int)Long.parseLong(treat_dy.substring(0, 4));

        boolean discardData = false;

        double[] ramtData = new double[6];
        for(int i = 0; i < ramtData.length; i++) {
            ramtData[i] = loan_ramt[i + (treat_year - 2004) + 1];
            if(i - 1 >= 0 && ramtData[i - 1] < ramtData[i]) {
                discardData = true;
            }
        }

        if(loan_ramt_str_sum.equals("")) {
            discardData = true;
        }
        /*
        if(loan_rat <= 0.0) {
            discardData = true;
        }
        */

        double[] labelData = new double[numOfOutputs];
        labelData[0] = ramtData[5] / loan_amt;
        labelData[1] = 1.0 - labelData[0];

        INDArray feature = Nd4j.create(featureData, new int[]{1, numOfInputs});
        INDArray label = Nd4j.create(labelData, new int[]{1, numOfOutputs});

        DataSet ds;
        if(discardData) {
            // discard data
            ds = null;
        } else {
            sum_ramt += ramtData[5];
            sum_loan_amt += loan_amt;

            ds = new DataSet(feature, label);
        }

        // System.out.println("\n ds ");
        // System.out.println( ds);
        logOut.write("\n ds \n");
        logOut.write(ds + "\n");


        return ds;
    }

    public static void evaluateModel(MultiLayerNetwork model) throws Exception {

        for(int i = 0; i <= 10; i++) {

            double[] featureData = new double[numOfInputs];
            featureData[0] =rescaleAmt(50000000 * i, 0, 500000000);

            /*
            featureData[1] = 0.75;
            featureData[2] = 0.75;
            */

            INDArray feature = Nd4j.create(featureData, new int[]{1, numOfInputs});
            INDArray output = model.output(feature);
            if(cnt % 100 == 0) {
                System.out.print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> feature = " + feature);
                System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> output = " + output);
            }
            logOut.write(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> feature = " + feature);
            logOut.write(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> output = " + output + "\n");
        }



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