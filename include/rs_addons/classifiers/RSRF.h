//developed by: Rakib

#ifndef RSRF_HEADER
#define RSRF_HEADER

#include <iostream>
#include <string>

#include <uima/api.hpp>

#include <ros/package.h>


#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>


#include <robosherlock/types/all_types.h>
#include <robosherlock/scene_cas.h>
#include <robosherlock/utils/time.h>
#include <robosherlock/DrawingAnnotator.h>

#include <rs_addons/classifiers/RSClassifier.h>


class RSRF : public RSClassifier

{

public:

  RSRF();

  void trainModel(std::string train_matrix_name, std::string train_label_name, std::string trained_file_name);

  int predict_multi_class(cv::Mat sample, cv::AutoBuffer<int>& out_votes);

  void classify(std::string trained_file_name,std::string test_matrix_name, std::string test_label_name,std::string obj_classInDouble);

  void classifyOnLiveData(std::string trained_file_name_saved, cv::Mat test_mat, double &det, double &confi);

  void annotate_hypotheses(uima::CAS &tcas, std::string class_name, std::string feature_name, rs::ObjectHypothesis &cluster, std::string set_mode, double &confi);

  ~ RSRF();

};

#endif
