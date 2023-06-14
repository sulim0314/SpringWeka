package ex03;

import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

public class Weka03LogisticRegression {
	
	String upDir="C:\\Weka-3-9\\data\\UniveralBank\\";
	
	Instances data, train, test;
	Logistic model;
	
	public void loadArff(String file) throws Exception{
		DataSource ds=new DataSource(file);		
		data=ds.getDataSet();
		//로지스틱 회귀는 데이터를 정규화해야 한다. ==> 웨카에서는 필터로 제공하고 있다.
		Normalize normalize=new Normalize();
		//정규화 필터를 data에 적용시키자
		normalize.setInputFormat(data);				
		Instances newData=Filter.useFilter(data,normalize);
		//newData가 정규화된 데이터==>학습 데이터와 테스트 데이터로 분리하자
		train=newData.trainCV(10, 0,new Random(1));
		test=newData.testCV(10, 0);
		
		train.setClassIndex(train.numAttributes()-1);
		test.setClassIndex(test.numAttributes()-1);
		
	}
	public void generateModel_Evaluate() throws Exception{
		Evaluation eval=new Evaluation(train);
		model=new Logistic();
		
		eval.crossValidateModel(model, train, 10, new Random(1));
		//model런
		model.buildClassifier(train);
		eval.evaluateModel(model, test);
		System.out.println("---------------------------------");
		System.out.printf("전체 데이터 건수: %d 개%n", (int)eval.numInstances());
		System.out.printf("정분류 건수 : %d 개%n",(int)eval.correct());
		System.out.printf("정분류율    : %.2f%n",eval.pctCorrect());
		//학습 모델을 저장하자
		saveModel(upDir+"UniveralBank.model");		
	}
	public void testPredict(String file) throws Exception{
		//저장했던 학습 모델을 복원시키자
		Classifier model2=loadModel(upDir+"UniveralBank.model");
		
		Instances predData=new DataSource(file).getDataSet();
		predData.setClassIndex(predData.numAttributes()-1);
		System.out.println("실제 데이터 수: "+predData.numInstances()+"개");
		
		//실제 데이터로 예측,분류할 때도 정규화 필터를 적용해야 한다
		Normalize norm=new Normalize();
		norm.setInputFormat(predData);
		Instances newData=Filter.useFilter(predData, norm);
		
		//학습모델에 실제 데이터를 넣어 분류한 값을 확인해보자
		Logistic lmodel=null;
		if(model2!=null && model2 instanceof Logistic) {
			lmodel=(Logistic)model2;
		}
		for(int i=0;i<newData.numInstances();i++) {
			System.out.println("-----Data "+i+"------------");
			double pred=lmodel.classifyInstance(newData.instance(i));
			System.out.println("pred: "+pred);
			int k=(int)newData.instance(i).classValue();
			System.out.println("실제 데이터 값: "+newData.classAttribute().value(k));
			System.out.println("에측한 데이터값: "+newData.classAttribute().value((int)pred));
			
			double[] prob=lmodel.distributionForInstance(newData.instance(i));
			System.out.println("****확률 분포***********");
			System.out.printf("No : prob[0]=%f%n", prob[0]);
			System.out.printf("Yes: prob[1]=%f%n", prob[1]);
		}
	}//-------------------------------
	public void saveModel(String file) {
		try {
			SerializationHelper.write(file, model);
			System.out.println(file+"에 로지스틱 학습 모델 저장 완료!!");
		}catch(Exception e) {
			e.printStackTrace();
		}
	}//-------------------------------
	public Classifier loadModel(String file) {
		try {
			Classifier model2=(Classifier)SerializationHelper.read(file);
			return model2;
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}//-------------------------------
	public static void main(String[] args) throws Exception {
		String file1="C:\\Weka-3-9\\data\\UniveralBank\\UniversalBank_preprocess.arff";
		String file2="C:\\Weka-3-9\\data\\UniveralBank\\UniversalBank_realData.arff";
		
		Weka03LogisticRegression app=new Weka03LogisticRegression();
		app.loadArff(file1);
		//app.generateModel_Evaluate();
		app.testPredict(file2);
	}

}
