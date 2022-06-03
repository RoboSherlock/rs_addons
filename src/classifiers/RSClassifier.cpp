// Developed by: Rakib

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <map>
#include <thread>

#include <yaml-cpp/yaml.h>
#include <ros/package.h>

#include <boost/filesystem.hpp>

#include <opencv2/opencv.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>


#include <robosherlock/types/all_types.h>
#include <robosherlock/scene_cas.h>
#include <robosherlock/utils/time.h>

#include <rs_addons/classifiers/RSClassifier.h>

using namespace cv;

RSClassifier::RSClassifier()
{
}

void RSClassifier::setLabels(std::string file_name, std::vector<std::string> &my_annotation)
{
    std::string packagePath = ros::package::getPath("rs_resources") + "/";
    std::string pathToFile;

    if(boost::filesystem::exists(file_name)) {
        pathToFile = file_name;
    }
    else {
        pathToFile = packagePath + file_name;
    }
    //To check the resource path................................................

    if(!boost::filesystem::exists(pathToFile)) {
        outError(file_name << " file does not exist in path " << packagePath << " to read the object's class label." << std::endl);
    }
    else {
        std::ifstream file((pathToFile).c_str());

        std::string str;
        std::vector<std::string> split_str;

        while(std::getline(file, str)) {
            boost::split(split_str, str, boost::is_any_of(":"));
            my_annotation.push_back(split_str[0]);
        }
    }
}

void RSClassifier::getLabels(const std::string path,  std::map<std::string, double> &input_file)
{
    double class_label = 1;
    std::ifstream file(path.c_str());
    std::string str;

    while(std::getline(file, str)) {
        input_file[str] = class_label;
        class_label = class_label + 1;
    }
}

// To read the descriptors matrix and it's label from /rs_resources/objects_dataset/extractedFeat folder...........
void RSClassifier::readFeaturesFromFile(std::string data_file_path, std::string label_name,
                                        cv::Mat &des_matrix, cv::Mat &des_label)
{
    cv::FileStorage fs;
    std::string pathToDataFile;
    std::string packagePath = ros::package::getPath("rs_resources") + '/';
    if(boost::filesystem::exists(data_file_path)) {
        pathToDataFile = data_file_path;
    }
    else {
        pathToDataFile = packagePath + data_file_path;
    }

    if(!boost::filesystem::exists(pathToDataFile)) {
        outError(data_file_path << " does not exist; Needs to be a full path or relateive to rs_resources package; please check" << std::endl);
    }
    else {
        fs.open(pathToDataFile, cv::FileStorage::READ);
        fs["descriptors"] >> des_matrix;
        fs["label"] >> des_label;
    }
}

// To show the confusion matrix and accuracy result...........................
void RSClassifier::evaluation(std::vector<int> test_label, std::vector<int> predicted_label, std::string obj_classInDouble)
{
    std::map < std::string, double > object_label;
    std::string resourcePath;
    resourcePath = ros::package::getPath("rs_resources") + '/';
    std::string label_path = "objects_dataset/extractedFeat/" + obj_classInDouble + ".txt";

    if(!boost::filesystem::exists(resourcePath + label_path)) {
        outError(obj_classInDouble << " file does not exist in path " << resourcePath + label_path << std::endl);
    };

    //To read the object class names from rs_resources/object_dataset/objects.txt.......
    getLabels(resourcePath + label_path, object_label);

    //Declare the confusion matrix which takes test data label (test_label) and predicted_label or class as inputs.
    //It's size is defined by the number of classes.
    std::vector <std::vector<int> >confusion_matrix(object_label.size(), std::vector<int>(object_label.size(), 0));

    for(int i = 0; i < test_label.size(); i++) {
        confusion_matrix[test_label[i] - 1][predicted_label[i] - 1] = confusion_matrix[test_label[i] - 1][predicted_label[i] - 1] + 1;
    }

    //To show the results of confusion matrix .................................
    std::cout << "confusion_matrix:" << std::endl;
    for(int i = 0; i < object_label.size(); i++) {
        for(int j = 0; j < object_label.size(); j++) {
            std::cout << confusion_matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
    //calculation of classifier accuracy........................................
    double c = 0;
    for(int i = 0; i < object_label.size(); i++) {
        c = c + confusion_matrix[i][i];
    }
    double Accuracy = (c / test_label.size()) * 100;
    std::cout << "classifier Accuray:" << Accuracy << std::endl;
}

//To save trained model frmom path in rs_addons/trainedData....
std::string RSClassifier::saveTrained(std::string trained_file_name)
{
    std::string packagePath;
    std::string save_train = "trainedData/";
    std::string a;
    packagePath = ros::package::getPath("rs_addons") + '/';

    if(!boost::filesystem::exists(packagePath + save_train)) {
        boost::filesystem::create_directory(boost::filesystem::path(packagePath + save_train));
        //    outError("Folder called (trainedData) is not found to save or load the generated trained model. "
        //             " Please create the folder in rs_addons/ and name it as trainedData, then run the annotator again "<<std::endl);
    }
    else {
        a = packagePath + save_train + "classifier.xml";
    }
    return a;
}

//To load trained model frmom path in rs_addons/trainedData....
std::string RSClassifier::loadTrained(std::string trained_file_name)
{
    std::string packagePath;
    std::string save_train = "trainedData/";
    std::string a;
    packagePath = ros::package::getPath("rs_addons") + '/';

    if(!boost::filesystem::exists(packagePath + save_train + trained_file_name + ".xml")) {
        outError(trained_file_name << " trained Model file does not exist in path " << packagePath + save_train << std::endl);
    }
    else {
        a = packagePath + save_train + trained_file_name + ".xml";
    }

    return a;
}

void RSClassifier::drawCluster(cv::Mat input, cv::Rect rect, const std::string &l, double confidence)
{
    std::stringstream ss;
    ss<<l<<" "<<confidence;
    std::string label = ss.str();
    cv::rectangle(input, rect, CV_RGB(0, 255, 0), 2);
    int offset = 15;
    int baseLine;
    cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_PLAIN, 1.5, 2.0, &baseLine);
    cv::putText(input, label, cv::Point(rect.x + (rect.width - textSize.width) / 2, rect.y - offset - textSize.height), cv::FONT_HERSHEY_PLAIN, 1.5, CV_RGB(0, 0, 0), 2.0);
}

void  RSClassifier::processPCLFeature(std::string memory_name, std::string set_mode, std::string feature_use,
                                      std::vector<rs::ObjectHypothesis> clusters, RSClassifier *obj_VFH, cv::Mat &color, std::vector<std::string> models_label, uima::CAS &tcas)
{
    outInfo("Number of cluster:" << clusters.size() << std::endl);

    for(size_t i = 0; i < clusters.size(); ++i) {
        rs::ObjectHypothesis &cluster = clusters[i];
        std::vector<rs::PclFeature> features;
        cluster.annotations.filter(features);

        for(size_t j = 0; j < features.size(); ++j) {
            rs::PclFeature &feats = features[j];
            outInfo("type of feature:" << feats.feat_type() << std::endl);
            std::vector<float> featDescriptor = feats.feature();
            outInfo("Size after conversion:" << featDescriptor.size());
            cv::Mat test_mat(1, featDescriptor.size(), CV_32F);
            for(size_t k = 0; k < featDescriptor.size(); ++k) {
                test_mat.at<float>(0, k) = featDescriptor[k];
            }
            outInfo("number of elements in :" << i << std::endl);
            double classLabel;
            double confi;
            obj_VFH->classifyOnLiveData(memory_name, test_mat, classLabel, confi);
            int classLabelInInt = classLabel;
            std::string classLabelInString = models_label[classLabelInInt - 1];

            //To annotate the clusters..................
            annotate_hypotheses(tcas, classLabelInString, feature_use, cluster, set_mode, confi);

            //set roi on image
            rs::ImageROI image_roi = cluster.rois.get();
            cv::Rect rect;
            rs::conversion::from(image_roi.roi_hires.get(), rect);

            //Draw result on image...........
            obj_VFH->drawCluster(color, rect, classLabelInString);

            outInfo("calculation is done" << std::endl);
        }
    }
}

//the function process and classify RGB images, which run from a .bag file.
void  RSClassifier::processCaffeFeature(std::string memory_name, std::string set_mode, std::string feature_use,
                                        std::vector<rs::ObjectHypothesis> clusters,
                                        RSClassifier *obj_caffe, cv::Mat &color, std::vector<std::string> models_label, uima::CAS &tcas) {
    //clusters comming from RS pipeline............................
    outInfo("Number of cluster:" << clusters.size() << std::endl);

    std::vector <std::thread> threads;

    //multithreading the iterations for the clusters
    for (int i = 0; i < clusters.size(); ++i) {
        threads.push_back(std::thread(&RSClassifier::calculateCaffeFeature, this, memory_name, set_mode, feature_use, clusters, std::ref(obj_caffe),
                                      std::ref(color), models_label, std::ref(tcas), i));
    }
    for (auto &th: threads) {
        th.join();
    }
}

void  RSClassifier::calculateCaffeFeature(std::string memory_name, std::string set_mode, std::string feature_use,
                                          std::vector<rs::ObjectHypothesis> clusters,
                                          RSClassifier *obj_caffe, cv::Mat &color, std::vector<std::string> models_label, uima::CAS &tcas, int i) {


    rs::ObjectHypothesis &cluster = clusters[i];
    std::vector<rs::Features> features;
    cluster.annotations.filter(features);

    outInfo("feature size:" << features.size());

    for(size_t j = 0; j < features.size(); ++j) {
        rs::Features &feats = features[j];
        outInfo("type of feature:" << feats.descriptorType() << std::endl);
        outInfo("size of feature:" << feats.descriptors << std::endl);
        outInfo("size of source:" << feats.source() << std::endl);

        //variable to store caffe feature..........
        cv::Mat featDescriptor;
        double classLabel;
        double confi;

        if(feats.source() == "Caffe") {
            rs::conversion::from(feats.descriptors(), featDescriptor);
            outInfo("Size after conversion: " << featDescriptor.size() << i);

            //The function generate the prediction result................
            obj_caffe->classifyOnLiveData(memory_name, featDescriptor, classLabel, confi);

            //class label in integer, which is used as index of vector model_label.
            int classLabelInInt = classLabel;
            std::string classLabelInString = models_label[classLabelInInt - 1];

            //To annotate the clusters..................
            annotate_hypotheses(tcas, classLabelInString, feature_use, cluster, set_mode, confi);

            //set roi on image
            rs::ImageROI image_roi = cluster.rois.get();
            cv::Rect rect;
            rs::conversion::from(image_roi.roi_hires.get(), rect);

            //Draw result on image...........................
            obj_caffe->drawCluster(color, rect, classLabelInString);
        }
        outInfo("calculation is done for thread: " << i << std::endl);
    }
}

RSClassifier::~ RSClassifier()
{
}
