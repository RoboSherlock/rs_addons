// Developed by: Rakib

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <rs_addons/classifiers/RSSVM.h>

//....................................Support vector machine........................................
RSSVM::RSSVM()
{
}

void RSSVM::trainModel(std::string train_matrix_name, std::string train_label_name, std::string trained_file_name)
{
    cv::Mat train_matrix;
    cv::Mat train_label;
    readFeaturesFromFile(train_matrix_name, train_label_name, train_matrix, train_label);
    std::cout << "size of train matrix:" << train_matrix.size() << std::endl;
    std::cout << "size of train label:" << train_label.size() << std::endl;

    std::string pathToSaveModel = saveTrained(trained_file_name);

    if(!pathToSaveModel.empty()) {


        cv::Mat var_type = cv::Mat(train_matrix.cols + 1, 1, CV_8U);
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

    cv::Ptr<cv::ml::SVM> my_svm = cv::ml::SVM::create();

    my_svm->setType(cv::ml::SVM::Types::C_SVC);
    my_svm->setKernel(cv::ml::SVM::KernelTypes::LINEAR);
    my_svm->setDegree(0.0);
    my_svm->setGamma(0.0);
    my_svm->setCoef0(0.0);
    my_svm->setC(2);
    my_svm->setNu(0);
    my_svm->setP(0);
    my_svm->setClassWeights(cv::Mat());
    my_svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, 0.000001));
    outInfo("Actually training the classifier");
    my_svm->train(trainData);



        outInfo("Saving model to:"<<pathToSaveModel);
        //To save the trained data.............................
        my_svm->save((pathToSaveModel).c_str());
    }
}

void RSSVM::classify(std::string trained_file_name_saved,
                     std::string test_matrix_name, std::string test_label_name, std::string obj_classInDouble)
{
    cv::Mat test_matrix;
    cv::Mat test_label;

    //To load the test data and it's label.............................
    readFeaturesFromFile(test_matrix_name, test_label_name, test_matrix, test_label);
    std::cout << "size of test matrix :" << test_matrix.size() << std::endl;
    std::cout << "size of test label" << test_label.size() << std::endl;


    //load the trained model
  cv::Ptr<cv::ml::SVM> your_svm = cv::Algorithm::load<cv::ml::SVM>(cv::String(loadTrained(trained_file_name_saved)));

  //To count the support vector
  int in = your_svm->getSupportVectors().rows;

    std::cout << "The number of support vector:" << in << std::endl;

    //convert test label matrix into a vector
    std::vector<double> con_test_label;
    test_label.col(0).copyTo(con_test_label);

    //Container to hold the integer value of labels............................
    std::vector<int> actual_label;
    std::vector<int> predicted_label;

    //Loop to prdict the rsult............................
    for(int i = 0; i < test_label.rows; i++) {
        double res = your_svm->predict(test_matrix.row(i));
        int prediction = res;
        predicted_label.push_back(prediction);
        double lab = con_test_label[i];
        int actual_convert = lab;
        actual_label.push_back(actual_convert);
        //std::cout<<"predicted class is::"<<prediction <<std::endl;
    }
    std::cout << "Result of Support Vector Machine :" << std::endl;
    evaluation(actual_label, predicted_label, obj_classInDouble);
}

void RSSVM::classifyOnLiveData(std::string trained_file_name_saved, cv::Mat test_mat, double &det, double &confi)
{
    //To load the test data and it's label.............................
    std::cout << "size of test matrix :" << test_mat.size() << std::endl;

    //load the trained model
  cv::Ptr<cv::ml::SVM> your_svm = cv::Algorithm::load<cv::ml::SVM>(cv::String(loadTrained(trained_file_name_saved)));

  //To count the support vector
  int in = your_svm->getSupportVectors().rows;


    std::cout << "The number of support vector:" << in << std::endl;
    double res = your_svm->predict(test_mat);
    det = res;
    std::cout << "predicted class is :" << res << std::endl;
}

void RSSVM::annotate_hypotheses(uima::CAS &tcas, std::string class_name, std::string feature_name, rs::ObjectHypothesis &cluster, std::string set_mode, double &confi)
{
    rs::Classification classResult = rs::create<rs::Classification>(tcas);
    classResult.classname.set(class_name);
    classResult.classifier("Support Vector Machine");
    classResult.featurename(feature_name);

    if(feature_name == "CNN") {
        classResult.classification_type("INSTANCE");
    }
    else if(feature_name == "VFH") {
        classResult.classification_type("SHAPE");
    }

    if(set_mode == "CL") {
        cluster.annotations.append(classResult);
    }
    else if(set_mode == "GT") {
        rs::GroundTruth setGT = rs::create<rs::GroundTruth>(tcas);
        setGT.classificationGT.set(classResult);
        cluster.annotations.append(setGT);
    }
    else {
        outError("You should set the parameter (set_mode) as CL or GT" << std::endl);
    }
}

RSSVM::~RSSVM()
{
}
