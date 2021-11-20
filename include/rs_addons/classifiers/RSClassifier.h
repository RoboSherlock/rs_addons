//developed by: Rakib

#ifndef RSCLASSIFIER_HEADER
#define RSCLASSIFIER_HEADER

#include <iostream>
#include <string>

#include <ros/package.h>
#include <boost/filesystem.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>


#include <robosherlock/types/all_types.h>
#include <uima/api.hpp>
#include <robosherlock/scene_cas.h>
#include <robosherlock/types/all_types.h>
#include <robosherlock/scene_cas.h>
#include <robosherlock/utils/time.h>
#include <robosherlock/DrawingAnnotator.h>

using namespace rs;

class RSClassifier
{
public:

  RSClassifier();
  
  virtual void trainModel(std::string train_matrix_name, std::string train_label_name, std::string trained_file_name)=0;
  
  virtual void classify(std::string trained_file_name,std::string test_matrix_name, std::string test_label_name, std::string obj_classInDouble)=0;

  virtual void classifyOnLiveData(std::string trained_file_name_saved, cv::Mat test_mat, double &det, double &confi)=0;

  virtual void annotate_hypotheses(uima::CAS &tcas,std::string class_name, std::string feature_name, ObjectHypothesis &cluster, std::string set_mode, double &confi)=0;

  void getLabels(const std::string path,  std::map<std::string, double> &input_file);

  void readFeaturesFromFile(std::string matrix_name, std::string label_name, cv::Mat &des_matrix, cv::Mat &des_label);
  
  //save model file
  std::string saveTrained(std::string trained_file_name);

  //load the model file
  std::string loadTrained(std::string trained_file_name);
  
  //some eval...what does it do?
  void evaluation(std::vector<int> test_label, std::vector<int> predicted_label,std::string obj_classInDouble);

  //probably draws a cluster on the image
  void drawCluster(cv::Mat input , cv::Rect rect, const std::string &label);

  //what is this?
  void  processPCLFeature(std::string memory_name,std::string set_mode, std::string feature_use,
                          std::vector<ObjectHypothesis> clusters, RSClassifier* obj_VFH , cv::Mat &color, std::vector<std::string> models_label , uima::CAS &tcas);

  //what about this?
  void  processCaffeFeature(std::string memory_name, std::string set_mode, std::string feature_use, std::vector<ObjectHypothesis> clusters, RSClassifier* obj_caffe ,
                            cv::Mat &color, std::vector<std::string> models_label, uima::CAS &tcas );

  void setLabels(std::string file_name, std::vector<std::string> &my_annotation);

  ~ RSClassifier();

};

#endif
