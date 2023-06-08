package ex02;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

//1. 데이터 로드-로드하여 훈련데이터셋과 검증데이터셋으로 분할한다
//2. 훈련과 검증데이터셋에서 정답 데이터 지정
//3. 교차검증 셋팅
//4. 알고리즘 모델을 생성해서 학습을 진행
//5. 결과 평가/검증
//6. 선택한 모델을 파일로 저장==> 후에 다시 모델을 로드해서 다시 학습을 진행
public class Weka02DecisionTree {
	static String file1="C:\\Weka-3-9\\data\\UniveralBank\\UniversalBank_preprocess.arff";
	static String file2="C:\\Weka-3-9\\data\\UniveralBank\\UniversalBank_realData.arff";
	Instances data, train, test;
	//Classifier model;//J48
	J48 model;
	
	public Weka02DecisionTree() {
		try {
			data=new Instances(new BufferedReader(new FileReader(file1)));
			data.randomize(new Random(1));
			train=data.trainCV(10, 0, new Random(1));
			test=data.testCV(10,0);
			
			train.setClassIndex(train.numAttributes()-1);//마지막 속성을 target(label)으로 지정
			test.setClassIndex(test.numAttributes()-1);
			
		}catch(Exception e) {
			e.printStackTrace();
		}
	}//--------------------------
	private void generateModel_Evaluate() {
		//모델 생성
		model=new J48();//Decision Tree 모델
		
		model.setMinNumObj(10);//노드 당 최소 인스턴스 수 지정=> 설정값이 클수록 노드수가 줄어들면서 나무의 깊이도 낮아짐
		model.setUnpruned(false);//가지치기 하려면 false값을 지정한다
		
		try {
			Evaluation eval=new Evaluation(train);
			eval.crossValidateModel(model, train, 10, new Random(1));//교차검증 설정
			model.buildClassifier(train);//학습 진행
			
			eval.evaluateModel(model, test);
			int nums=(int)eval.numInstances();//분류 대상 데이터수
			int accuracy=(int)eval.correct();//정분류 건수
			int percent =accuracy*100/nums;//정분류율
			System.out.println("분류대상 데이터수: "+nums);
			System.out.println("정분류 건수: "+accuracy);
			System.out.println("정분류율1: "+percent+"%");
			System.out.println("정분류율2: "+String.format("%.2f", eval.pctCorrect())+"%");
			System.out.println("--------------------------");
			System.out.println(eval.toSummaryString());
			
			this.treeViewInstances(data, model, eval);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}//--------------------------
	//Weka제공 시각화
	private void treeViewInstances(Instances data2, J48 model, Evaluation eval) throws Exception {
		
		JTextArea ta=new JTextArea(eval.toSummaryString());
		JScrollPane sp=new JScrollPane(ta);
		
		TreeVisualizer panel=new TreeVisualizer(null, model.graph(), new PlaceNode2());
		
		JFrame f=new JFrame("정분류율: "+String.format("%.2f", eval.pctCorrect()));
		f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		f.getContentPane().add(panel,"Center");
		f.getContentPane().add(sp,"South");
		
		f.setSize(800,800);
		f.setVisible(true);
		//panel.fitToScreen();
		
		
	}//--------------------------
	private void testPredict(String file) {
		try {
			DataSource ds=new DataSource(file);			
			Instances predData=ds.getDataSet();
			predData.setClassIndex(predData.numAttributes()-1);
			System.out.println("데이터 수: "+predData.numInstances());
			
			for(int i=0;i<predData.numInstances();i++) {
				System.out.println("------Data "+i+"---------------");
				double pred=model.classifyInstance(predData.instance(i));
				System.out.println("pred: "+pred);
				System.out.println("given value: "+predData.classAttribute().value((int)predData.instance(i).classValue()));
				System.out.println("predicted value: "+predData.classAttribute().value((int)pred));
				
				double[] prediction=model.distributionForInstance(predData.get(i));
				double prob1=prediction[0];
				double prob2=prediction[1];
				System.out.println("****확률 분포*****************");
				System.out.println(prob1);
				System.out.println(prob2);				
			}
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}//-------------------------------
	public static void main(String[] args) {
		
		Weka02DecisionTree app=new Weka02DecisionTree();
		app.generateModel_Evaluate();
		
		
		app.testPredict(file2);
	}//--------------------------

}//////////////////////////////////////////////////////////////
