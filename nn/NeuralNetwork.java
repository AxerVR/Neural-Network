package nn.controller;

import helper.MyMath;
import java.util.List;

public class NeuralNetwork {

    MyMath weights_ih, weights_ho, bias_h, bias_o;
    double l_rate = 0.01;

    public NeuralNetwork(int i, int h, int o) {
        weights_ih = new MyMath(h, i);
        weights_ho = new MyMath(o, h);

        bias_h = new MyMath(h,1);
        bias_o = new MyMath(o,1);

    }

    public List<Double> classify(double[] A)
    {
        MyMath input = MyMath.fromArray(A);
        MyMath hidden = MyMath.multiply(weights_ih, input);
        hidden.add(bias_h);
        hidden.sigmoid();

        MyMath output = MyMath.multiply(weights_ho, hidden);
        output.add(bias_o);
        output.sigmoid();

        return output.toArray();
    }


    public void adjust(double[][]X, double[][]Y, int epochs)
    {
        for(int i = 0; i < epochs; i++)
        {
            int sampleN =  (int)(Math.random() * X.length);
            this.train(X[sampleN], Y[sampleN]);
        }
    }

    public void train(double [] X,double [] Y)
    {
        MyMath input = MyMath.fromArray(X);
        MyMath hidden = MyMath.multiply(weights_ih, input);
        hidden.add(bias_h);
        hidden.sigmoid();

        MyMath output = MyMath.multiply(weights_ho,hidden);
        output.add(bias_o);
        output.sigmoid();

        MyMath target = MyMath.fromArray(Y);

        MyMath error = MyMath.subtract(target, output);
        MyMath gradient = output.deriveSigmoid();
        gradient.multiply(error);
        gradient.multiply(l_rate);

        MyMath hidden_T = MyMath.transpose(hidden);
        MyMath who_delta =  MyMath.multiply(gradient, hidden_T);

        weights_ho.add(who_delta);
        bias_o.add(gradient);

        MyMath who_T = MyMath.transpose(weights_ho);
        MyMath hidden_errors = MyMath.multiply(who_T, error);

        MyMath h_gradient = hidden.deriveSigmoid();
        h_gradient.multiply(hidden_errors);
        h_gradient.multiply(l_rate);

        MyMath i_T = MyMath.transpose(input);
        MyMath wih_delta = MyMath.multiply(h_gradient, i_T);

        weights_ih.add(wih_delta);
        bias_h.add(h_gradient);

    }


}
