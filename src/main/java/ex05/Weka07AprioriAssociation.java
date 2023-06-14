package ex05;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import weka.associations.Apriori;
import weka.associations.AssociationRule;
import weka.associations.AssociationRules;
import weka.associations.Item;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.OneR;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;

public class Weka07AprioriAssociation {
	
	String file="C:\\Weka-3-9\\data\\Book\\CharlesBookClub_preprocess.arff";
	DataSource ds;
	Instances data;
	Apriori model;//비지도학습-연관분석 모델
	
	public void loadArff(String file) throws Exception{
		ds=new DataSource(file);
		data=ds.getDataSet();
	}
	//연관규칙 알아내기
	public void association() throws Exception{
		model=new Apriori();
		model.setLowerBoundMinSupport(0.05);//지지도(Support) 설정
		//전체 4000건 중 5%이상 거래가 이뤄진 데이터를 대상으로 설정
		model.setMetricType(new SelectedTag(1, model.TAGS_SELECTION));
		//metricType를 향상도(Lift)로 지정 (디폴트값은 신뢰도(Confidence))
		//신뢰도:0, 향상도:1
		model.setMinMetric(1.5);
		//향상도 최소값을 1.5로 지정한다
		//A를 구했을때 B를 함께 구매할 비율이 1.5배 이상 나타난다
		model.setNumRules(10);
		//학습 진행
		model.buildAssociations(data);
		//evaluate은 필요 없고, 연관규칙 추출을 해야한다
		AssociationRules rules=model.getAssociationRules();
		List<AssociationRule> rule_list=rules.getRules();
		printRule(rule_list);
		
		//전조현상 A와 병행현상 B에서 발생한 모든 속성값별 발생횟수를 계산해보자
		Map<String, Integer> attrNameCounts=countByItemSets(rule_list);
		//System.out.println(attrNameCounts);
		
		//데이터의 속성명 저장해서 List로 반환하는 메서드
		List<String> attrNames=indexOfInstanceList(data);
		//System.out.println(attrNames);
		
		//최다 발생하는 아이템을 구하는 메서드
		int topIndex=fetchTopAttribute(attrNames, attrNameCounts);

		//OneR분류 알고리즘으로 최다 발생 속성과 연관속성의 밀접도를 확인해보자
		buildOneR(topIndex);
		
	}//----------------------------------------
	
	private void buildOneR(int topIndex) throws Exception {
		System.out.println("----OneR build---------------");
		data.setClassIndex(topIndex);
		System.out.println(data.classIndex()+", "+data.classAttribute());
		
		Instances train=data.trainCV(10, 0, new Random(1));
		Instances test=data.testCV(10, 0);
		
		Evaluation eval=new Evaluation(train);
		
		OneR model=new OneR();
		eval.crossValidateModel(model, train, 10, new Random(1));
		model.buildClassifier(train);//학습
		eval.evaluateModel(model, test);
		System.out.println("분류대상 데이터 건수: "+(int) eval.numInstances()+"개");
		System.out.println("정분류 건수: "+ (int)eval.correct()+"개");
		System.out.println("정분류율: "+String.format("%.2f", eval.pctCorrect())+"%");
		System.out.println("----------------------------");
	}//----------------------------------------
	//전조현상 A와 병행현상 B에서 발생한 모든 속성값별 발생횟수를 계산해보자
	private Map<String, Integer> countByItemSets(List<AssociationRule> rule_list) {
		Map<String, Integer> map=new HashMap<>();
		for(AssociationRule ar:rule_list) {
			Collection<Item> premise=ar.getPremise();
			//전조현상
			map=countByAttribute(premise, map);
			//병행현상
			Collection<Item> consequence=ar.getConsequence();
			map=countByAttribute(consequence,map);
		}
		return map;
	}//----------------------------------------
	
	private Map<String, Integer> countByAttribute(Collection<Item> itemSet, Map<String, Integer> map) {
		for(Item item:itemSet) {
			//속성명 추출
			String attrName=item.getAttribute().name();//속성명
			String yn=item.getItemValueAsString();//y,n 속성값
			
			//속성명 발생회수 저장
			if(map.get(attrName)==null) {
				map.put(attrName, 1);
			}else {
				Integer val=map.get(attrName);
				map.put(attrName, val+1);
			}
		}
		return map;
	}//----------------------------------------
	private List<String> indexOfInstanceList(Instances data) {
		List<String> attrNames=new ArrayList<>();
		Instance obj=data.firstInstance();
		for(int i=0;i<obj.numAttributes();i++) {
			Attribute attr=obj.attribute(i);
			attrNames.add(attr.name());
		}
		return attrNames;
	}//----------------------------------------
	private int fetchTopAttribute(List<String> attrNames, Map<String, Integer> attrNameCounts) {
		String topAttrName="";
		int topCount=0;
		int topIndex=0;
		
		for(int i=0;i<attrNames.size();i++) {
			String currAttrName=attrNames.get(i);
			if(currAttrName!=null) {
				Integer cnt=0;
				//System.out.println("currAttrName="+currAttrName);
				cnt=attrNameCounts.get(currAttrName);
				if(cnt==null) continue;
				if(cnt>topCount) {
					topCount=cnt;
					topAttrName=currAttrName;
					topIndex=i;
				}				
			}//if----			
		}//for-----------------
		System.out.println("최다 발생 속성명: "+topAttrName+"="+topCount+", index: "+topIndex);
		
		return topIndex;
	}//----------------------------------------
	
	
	
	
	public void printRule(List<AssociationRule> rule_list) throws Exception{
		int i=1;
		for(AssociationRule ar:rule_list) {
			System.out.println("****"+i++ +"*************");
			System.out.println(ar);
			double[] metric=ar.getMetricValuesForRule();
			System.out.println("신뢰도(Confidence): "+metric[0]);
			System.out.println("향상도(Lift)      : "+metric[1]);
			System.out.println("전조현상 A["+ar.getPremise()+"]에 대한 지지도: "+ar.getPremiseSupport());
			System.out.println("병행현상 B["+ar.getConsequence()+"]에 대한 지지도: "+ar.getTotalSupport());
			//premise +consequence
			System.out.println("전체지지도: "+ar.getConsequenceSupport());
		}//for----
	}

	public static void main(String[] args) throws Exception{
		Weka07AprioriAssociation app=new Weka07AprioriAssociation();
		app.loadArff(app.file);
		app.association();

	}

}
