package ex03;

import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
//데이터 얻기:  https://learnersdesk.weebly.com/weka-tutorials.html
public class Weka04LinearRegression {
	DataSource ds;
	Instances data;
	LinearRegression model;
	
	public void loadArff(String file) throws Exception{
		ds=new DataSource(file);
		data=ds.getDataSet();
		//종속변수(y)-target 설정
		data.setClassIndex(data.numAttributes()-1);
	}
	public void generateModel() throws Exception{
		model=new LinearRegression();
		model.buildClassifier(data);
		System.out.println("model공식: "+model);
	}
	public void predictHouse() throws Exception{
		//첫번째 집의 판매가격을 예측해보자
		//Instance firstHouse=data.firstInstance();
		//Instance lastHouse=data.lastInstance();//마지막 집
		Instance House=data.instance(3);//4번째 집
		
		double predictPrice=model.classifyInstance(House);
		System.out.println("--------------------------");
		System.out.println("집의 예측 판매가격: "+predictPrice);
		System.out.println("--------------------------");
		
	}
	public void predictHouse(String file) throws Exception{
		DataSource ds2=new DataSource(file);
		Instances mydata=ds2.getDataSet();
		mydata.setClassIndex(mydata.numAttributes()-1);
		Instance myhouse=mydata.instance(0);
		
		
		System.out.println("실제 가격: "+myhouse.classValue());
		double price=model.classifyInstance(myhouse);
		System.out.println("예측 가격: "+price);
	}
	public double predictCalc(double houseSize, double lotSize, int bedrooms, int bathroom) {
		double sellPrice=-26.6882 * houseSize +    7.0551 * lotSize +   43166.0767 * bedrooms +   42292.0901 * bathroom + -21661.1208;
		return sellPrice;
	}
	
	public static void main(String[] args) throws Exception {
		Weka04LinearRegression app=new Weka04LinearRegression();
		String file="C:\\Weka-3-9\\data\\House\\house.arff";
		String file2="C:\\Weka-3-9\\data\\House\\myhouse.arff";
		app.loadArff(file);
		app.generateModel();
		app.predictHouse();
		System.out.println("----MyHouse 판매 예측 가격-----------");
		app.predictHouse(file2);
		System.out.println("-------------------------------");
		double price=app.predictCalc(5500,1025,7,3);
		System.out.println("price: "+price);
	}

}
