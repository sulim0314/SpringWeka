package ex01;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

//NaiveBayes알고리즘을 이용해서 iris데이터를 분류하는 학습을 해보자. (다중 분류)
//1. 데이터 로드-로드하여 훈련데이터셋과 검증데이터셋으로 분할한다
//2. 훈련과 검증데이터셋에서 정답 데이터 지정
//3. 교차검증 셋팅
//4. 알고리즘 모델을 생성해서 학습을 진행
//5. 결과 평가/검증
//6. 선택한 모델을 파일로 저장==> 후에 다시 모델을 로드해서 다시 학습을 진행
public class Weka01NaiveBayes {

	Instances irisData, train, test;
	NaiveBayes model; // 알고리즘 모델
	DataSource ds;

	// 1. 데이터 로드
	public void loadArff(String path) {
		try {
			ds = new DataSource(path);
			irisData = ds.getDataSet();
			//iris데이터셋을 훈련데이터와 검증(테스트)데이터로 분류
			irisData.randomize(new Random(1));//데이터셋을 랜덤하게 섞는다.
			train = irisData.trainCV(10, 0, new Random(1));//학습데이터
			test = irisData.testCV(10, 0);
			
			//정답 데이터 지정 => class assigner
			if(irisData.classIndex() == -1) {
				train.setClassIndex(train.numAttributes()-1);//마지막 속성을 정답 데이터(target, class, label)로 지정
				test.setClassIndex(test.numAttributes()-1);
			}
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	// 2. 모델 생성
	public void generateModel() {
		model = new NaiveBayes();
	}//----------------------------

	private void evaluate(int numfolds) {
		//훈련데이터에 교차 검증 셋팅
		Evaluation eval = null;
		try {
			eval = new Evaluation(train);
			eval.crossValidateModel(model, train, numfolds, new Random(1));
			
			//모델을 이용해서 학습시키기
			model.buildClassifier(train);
			
			//test데이터로 평가하기
			eval.evaluateModel(model, test);
			
			//결과 출력
			System.out.println("정분류율: "+ String.format("%.2f", eval.pctCorrect()));
			
			String result = eval.toSummaryString();
			System.out.println("============================");
			System.out.println(result);
			System.out.println("============================");
			
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}//----------------------------

	private void testPredict(String path_test) {
		try {
			DataSource ds = new DataSource(path_test);
			Instances predictData = ds.getDataSet();
			predictData.setClassIndex(predictData.numAttributes()-1);
			System.out.println("실제 데이터 수: "+ predictData.numInstances());
			for(int i=0; i<predictData.numInstances(); i++) {
				System.out.println("------Data "+ i +"------------");
				Instance inc = predictData.instance(i);
				double pred = model.classifyInstance(inc);
				System.out.println("pred(예측값) : "+ pred);
				System.out.println("predicted value: "+ predictData.classAttribute().value((int)pred));
				//분류 결과를 문자열로 반환
			}
			
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		
	}//----------------------------
	
	public static void main(String[] args) {
		String path = "C:/Weka-3-9/data/iris.arff";// 훈련 및 검증에 사용
		String path_test = "C:/Weka-3-9/data/iris_test.arff"; // 훈련된 모델을 이용해 임의의 테스트 데이터를 분류할 때 사용

		Weka01NaiveBayes app = new Weka01NaiveBayes();
		app.loadArff(path);
		app.generateModel();
		app.evaluate(10); // 교차검증수 10개
		app.testPredict(path_test); // 훈련모델을 테스트해보자
		
	
	}


}
