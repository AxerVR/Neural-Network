package nn.agent;

import jade.core.Agent;
import jade.core.behaviours.OneShotBehaviour;
import java.util.List;
import nn.controller.*;

public class NNAgent extends Agent {
	
	private NNGui myGui;
	
	static double [][] X = new double[][]{
            {0,0,1,0,0,0,1,1,0,0,1,1,1,1,1,0,1,1,0,1,0,0,1,0,1},//IZQUIERDA
            {0,0,1,0,0,0,1,1,0,0,1,1,1,1,1,0,1,1,0,1,0,0,1,0,1},//IZQUIERDA
            {0,0,1,0,0,0,1,0,0,0,1,1,1,1,1,0,1,0,0,1,0,0,1,0,1},//IZQUIERDA
            {0,1,1,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0,0,1,0,1,1,0,1},//IZQUIERDA
            {0,0,1,0,0,0,0,1,1,0,1,1,1,1,1,1,0,1,1,0,1,0,1,0,0},//DERECHA
            {0,0,1,0,0,0,0,0,1,0,1,1,1,1,1,1,0,0,1,0,1,0,1,0,0},//DERECHA
            {0,0,1,1,0,0,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,0,1,1,0}//DERECHA
    };
    static double [][] Y = new double[][]{
            {0,0},{0,0},{0,0},{0,0},{1,1},{1,1},{1,1}
    };
	
	//Traffic signs test data
/*	double [][] input = {
			{0,0,1,1,0,0,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,0},//DERECHA
			{1,1,0,0,0,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0}//IZQUIERDA
	};*/

  protected void setup() {
	  
	myGui = new NNGui(this);
	myGui.showGui();
  }
  
  // Put agent clean-up operations here
	protected void takeDown() {
		// Close the GUI
		myGui.dispose();
		// Printout a dismissal message
		System.out.println("NeuralNetwork-agent " + getAID().getName() + " terminating.");
	}
  
    public void btnAction(String inputString) {
		addBehaviour(new OneShotBehaviour() {
			public void action() {
				
				String[] userRawValues = inputString.split(",");

				double [] input = new double [userRawValues.length];

				for (int i = 0; i < userRawValues.length; i++) {
					input[i] = Double.parseDouble(userRawValues[i]);
				}

				NeuralNetwork nn = new NeuralNetwork(25,12,2);


				List<Double>output;

				nn.adjust(X, Y, 50000);

				output = nn.classify(input);
				System.out.println(output.toString());
		
			}
		} );
	}
}
