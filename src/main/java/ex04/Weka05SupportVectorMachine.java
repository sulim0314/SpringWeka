package ex04;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

//functions/SMO ==>서포트벡터머신 알고리즘 (주의. 데이터가 정규화 또는 표준화가 되어 있어야 함)
public class Weka05SupportVectorMachine {
	
	String file="C:\\Weka-3-9\\data\\Titanic\\titanic_ko_remove_reorder_normalize.arff";
	Instances data, train, test;
	SMO model;
	
	public void loadArff(String file) throws Exception{
		data=new DataSource(file).getDataSet();
		data.randomize(new Random(1));
		
		//Reorder reorder=new Reorder();
		//reorder.setAttributeIndices("1,3-last,2"); => 이미 reorder된 파일로 하므로 필요 없음
		
		//정규화하기 ==>이미 다 전처리함
		//Normalize norm=new Normalize();
		//norm.setInputFormat(data);
		//Instances newData=Filter.useFilter(data,norm)
		
		train=data.trainCV(10, 0, new Random(1));
		test=data.testCV(10, 0);
		
		//정답데이터 지정
		//train.setClassIndex(1); 2번째 속성(survived)을 정답데이터로 지정하는 경우
		//==>reorder했으므로 마지막 속성을 정답데이터로 지정한다
		train.setClassIndex(train.numAttributes()-1);
		test.setClassIndex(test.numAttributes()-1);
		
	}//-----------------------------
	public void generateModel_Evaluate() throws Exception{
		Evaluation eval=new Evaluation(train);
		model=new SMO();
		eval.crossValidateModel(model, train, 10, new Random(1));
		System.out.println("---SMO model run before------------");
		//학습 진행
		model.buildClassifier(train);
		System.out.println("---SMO model run after------------");
		
		//검증
		eval.evaluateModel(model, test);
		System.out.println("-----------------------------");
		System.out.printf("전체 데이터 건수: %d개%n",(int)eval.numInstances());
		System.out.printf("정 분류 건수: %d개%n",(int)eval.correct());
		System.out.println("정분류율 : "+String.format("%.2f", eval.pctCorrect())+"%");
		System.out.println("------------------------------");
		System.out.println(eval.toSummaryString());
	}//-----------------------------
	
	public void predict(String file) throws Exception{
		
	}//-----------------------------
	public static void main(String[] args) throws Exception {
		Weka05SupportVectorMachine app=new Weka05SupportVectorMachine();
		app.loadArff(app.file);
		app.generateModel_Evaluate();

	}

}
