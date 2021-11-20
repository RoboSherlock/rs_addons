// Developed by: Rakib

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <rs_addons/classifiers/RSRF.h>

using namespace cv;

//..............................Random Forest Classifier.........................................
RSRF::RSRF()
{

}

void RSRF::trainModel(std::string train_matrix_name, std::string train_label_name, std::string trained_file_name)
{
    cv::Mat train_matrix;
    cv::Mat train_label;
    readFeaturesFromFile(train_matrix_name, train_label_name, train_matrix, train_label);
    std::cout << "size of train matrix:" << train_matrix.size() << std::endl;
    std::cout << "size of train label:" << train_label.size() << std::endl;

    std::string pathToSaveModel= saveTrained(trained_file_name);

    if(!pathToSaveModel.empty())
    {

        cv::Mat var_type = cv::Mat(train_matrix.cols +1, 1, CV_8U);
    var_type.setTo(cv::Scalar(cv::ml::VAR_NUMERICAL));
    var_type.at<uchar>(train_matrix.cols, 0) = cv::ml::VAR_CATEGORICAL;

    cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(train_matrix,  //samples
                                                                 cv::ml::ROW_SAMPLE, //layout
                                                                 train_label, //responses
                                                                 cv::noArray(), //varIdx
                                                                 cv::noArray(), //sampleIdx
                                                                 cv::noArray(), //sampleWeights
                                                                 var_type //varType
                                                                 );

    cv::Ptr<cv::ml::RTrees> rtree = cv::ml::RTrees::create();

    rtree->setMaxDepth(25);
    rtree->setMinSampleCount(10);
    rtree->setRegressionAccuracy(0);
    rtree->setUseSurrogates(false);
    rtree->setMaxCategories(15);
    rtree->setPriors(cv::Mat());
    rtree->setCalculateVarImportance(false);
    //TODO: number of variables randomly selected at node and used to find the best split(s) is missing
    //rtree->setCVFolds(4);
    rtree->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 100, 0.01f)); //termination criteria, max # of trees, forest accuracy

    rtree->train(trainData);
    //rtree->train(train_matrix, cv::ml::ROW_SAMPLE, train_label);



        //To save the trained data.............................
        rtree->save((pathToSaveModel).c_str());
    }
}

//int RSRF::predict_multi_class(cv::Mat sample, cv::AutoBuffer<int>& out_votes)
//{
//#if CV_MAJOR_VERSION == 2
//int result = 0;
//int k;

////TODO: find out what nclassses and ntrees is in OpenCV3
//if( nclasses > 0 ) //classification
//{

//    int* votes = out_votes;
//    std::cout<<"number of trees:"<<ntrees<<std::endl;
//    std::cout<<"number of n-classes:"<<nclasses<<std::endl;
//    std::cout<<"size of memory:"<<sizeof(*votes)<<std::endl;
//    memset( votes, 0, sizeof(*votes)*nclasses );
//    for( k = 0; k < ntrees; k++ )
//    {
//      CvDTreeNode* predicted_node = trees[k]->predict( sample, cv::Mat());
//      int nvotes;
//      int class_idx = predicted_node->class_idx;
//      //double val = predicted_node->value;
//      //std::cout<<"class label:"<<val<<std::endl;
//      CV_Assert( 0 <= class_idx && class_idx < nclasses );

//      nvotes = ++votes[class_idx];

//      std::cout<<"class index:"<< class_idx <<std::endl;

//    }

//    result = ntrees;
//}

//return result;
//#elif CV_MAJOR_VERSION == 3
//  return 0;//cv::ml::RTrees::getRoots().size();
//#endif

//}

void RSRF::classify(std::string trained_file_name_saved, std::string test_matrix_name, std::string test_label_name, std::string obj_classInDouble)
{
    cv::Mat test_matrix;
    cv::Mat test_label;
    readFeaturesFromFile(test_matrix_name, test_label_name, test_matrix, test_label);
    std::cout << "size of test matrix :" << test_matrix.size() << std::endl;
    std::cout << "size of test label" << test_label.size() << std::endl;


    //To load the trained data

    cv::Ptr<cv::ml::RTrees> rtree = cv::Algorithm::load<cv::ml::RTrees>(cv::String(loadTrained(trained_file_name_saved)));


    //convert test label matrix into a vector
    std::vector<double> con_test_label;
    test_label.col(0).copyTo(con_test_label);

    //Container to hold the integer value of labels
    std::vector<int> actual_label;
    std::vector<int> predicted_label;
//  cv::AutoBuffer<int> out_votes;

    for(int i = 0; i < test_label.rows; i++)
    {

        double res = rtree->predict(test_matrix.row(i));

        int prediction = res;

//      int class_index=res-1;
//      double numberOfTrees=predict_multi_class(test_matrix.row(i), out_votes);

//      std::cout<<"number of trees: "<<numberOfTrees<<std::endl;
//      std::cout<<"class : "<<res <<" gets : "<<out_votes[class_index]<<" votes"<<std::endl;
//      std::cout<<"confidence of the RF Classifier: "<<out_votes[class_index]/numberOfTrees<<std::endl;

        predicted_label.push_back(prediction);
        double lab = con_test_label[i];
        int actual_convert = lab;
        actual_label.push_back(actual_convert);
    }
    std::cout << "Random forest Result :" << std::endl;
    evaluation(actual_label, predicted_label, obj_classInDouble);
}

void RSRF::classifyOnLiveData(std::string trained_file_name_saved, cv::Mat test_mat, double &det, double &confi)
{
    cv::AutoBuffer<int> out_votes;
    //To load the test data.............................
    std::cout << "size of test matrix :" << test_mat.size() << std::endl;

    //To load the trained data................................

    cv::Ptr<cv::ml::RTrees> rtree = cv::Algorithm::load<cv::ml::RTrees>(cv::String(loadTrained(trained_file_name_saved)));

     double res = rtree->predict(test_mat);


    int class_index=res-1;
//     double numberOfTrees=predict_multi_class(test_mat, out_votes);
//     double con=out_votes[class_index]/numberOfTrees;
//     std::cout<<"number of trees: "<<numberOfTrees<<std::endl;
    std::cout<<"class : "<<res <<" gets : "<<out_votes[class_index]<<" votes"<<std::endl;
//     std::cout<<"confidence of the RF Classifier: "<<con<<std::endl;
    det = res;
//     confi=con;
}

void RSRF::annotate_hypotheses(uima::CAS &tcas, std::string class_name, std::string feature_name, rs::ObjectHypothesis &cluster, std::string set_mode, double &confi)
{
    rs::ClassConfidence conResult = rs::create<rs::ClassConfidence>(tcas);
    conResult.score.set(confi);
    conResult.source("Random Forest");
    cluster.annotations.append(conResult);

    rs::Classification classResult = rs::create<rs::Classification>(tcas);
    classResult.classname.set(class_name);
    classResult.classifier("Random Forest");
    classResult.featurename(feature_name);

    if(feature_name == "CNN")
    {
        classResult.classification_type("INSTANCE");
    } else if(feature_name == "VFH")
    {
        classResult.classification_type("SHAPE");
    }

    if(set_mode == "CL")
    {
        cluster.annotations.append(classResult);
    }
    else if(set_mode == "GT")
    {
        rs::GroundTruth setGT = rs::create<rs::GroundTruth>(tcas);
        setGT.classificationGT.set(classResult);
        cluster.annotations.append(setGT);

    }
    else
    {
        outError("You should set the parameter (set_mode) as CL or GT"<<std::endl);
    }
}

RSRF::~ RSRF()
{
}