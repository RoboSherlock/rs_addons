//developed by: Rakib

#ifndef RSKNN_HEADER
#define RSKNN_HEADER

#include <iostream>
#include <string>

#include <uima/api.hpp>

#include <ros/package.h>


#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

#include <robosherlock/scene_cas.h>
#include <robosherlock/types/all_types.h>
#include <robosherlock/scene_cas.h>
#include <robosherlock/utils/time.h>
#include <robosherlock/DrawingAnnotator.h>

#include <rs_addons/classifiers/RSClassifier.h>

class RSKNN : public RSClassifier
{

public:

  cv::Ptr<cv::ml::KNearest> knncalld;

  RSKNN(int K);

  void trainModel(std::string train_matrix_name, std::string train_label_name, std::string trained_file_name);

  void classify(std::string trained_file_name,std::string test_matrix_name, std::string test_label_name,std::string obj_classInDouble);

  void classifyOnLiveData(std::string trained_file_name_saved, cv::Mat test_mat, double &det, double &confi);

  void annotate_hypotheses (uima::CAS &tcas, std::string class_name, std::string feature_name, rs::ObjectHypothesis &cluster, std::string set_mode, double &confi);

  void classifyKNN(std::string train_matrix_name,std::string train_label_name,
                   std::string test_matrix_name, std::string test_label_name, std::string obj_classInDouble, int default_k);

  std::pair<double,double> classifyOnLiveDataKNN(cv::Mat test_mat);

  void processPCLFeatureKNN(std::string set_mode, std::string feature_use,
                            std::vector<rs::ObjectHypothesis> clusters, cv::Mat &color,std::vector<std::string> models_label, uima::CAS &tcas);

  void processCaffeFeatureKNN( std::string set_mode, std::string feature_use, std::vector<rs::ObjectHypothesis> clusters,
                               cv::Mat &color, std::vector<std::string> models_label, uima::CAS &tcas);

  void loadModelFile(std::string pathToModelFile);

  ~RSKNN();
private:

  cv::Mat trainingData_;
  cv::Mat dataLabels_;

};

#endif
