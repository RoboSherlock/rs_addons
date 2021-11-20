// Developed by: Rakib

#include <uima/api.hpp>
#include <iostream>
#include <vector>
#include <pcl/point_types.h>
#include <robosherlock/types/all_types.h>
#include <robosherlock/scene_cas.h>
#include <robosherlock/utils/time.h>


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>


#include <ros/package.h>
#include <boost/filesystem.hpp>
#include <rs_addons/classifiers/RSSVM.h>
#include <robosherlock/DrawingAnnotator.h>

using namespace uima;

class SvmAnnotator : public DrawingAnnotator
{
private:

  cv::Mat color;

  //set_mode should be GT(groundTruth) or CF (classify)
  std::string set_mode;

  //dataset_use should be IAI (kitchen data from IAI) or WU (data from Washington University) or BOTH...
  std::string dataset_use;

  //feature_use should be VFH, CVFH, CNN, VGG16 .....
  std::string feature_use;

  //the name of trained model ifle in folder rs_addons/trainedData
  std::string trained_model_name;

  //vector to hold split trained_model_name
  std::vector<std::string> split_model;

  //the name of actual_class_label map file in path rs_resources/objects_dataset/extractedFeat/
  std::string actual_class_label;

  //vector to hold classes name
  std::vector<std::string> model_labels;

public:

  SvmAnnotator(): DrawingAnnotator(__func__)
  {  }

  RSClassifier* svmObject = new RSSVM;

  TyErrorId initialize(AnnotatorContext &ctx)
  {
    outInfo("initialize");

    ctx.extractValue("set_mode", set_mode);
    ctx.extractValue("trained_model_name", trained_model_name);
    ctx.extractValue("actual_class_label", actual_class_label);

    outInfo("Name of the loaded files for SVM are:"<<std::endl);

    outInfo("set_mode:"<<set_mode<<std::endl);
    outInfo("trained_model_name:"<<trained_model_name<<std::endl);
    outInfo("actual_class_label:"<<actual_class_label<<std::endl);

    svmObject->setLabels(actual_class_label, model_labels);

    boost::split(split_model, trained_model_name, boost::is_any_of("_"));

    dataset_use= split_model[0];
    outInfo("dataset_use:"<<dataset_use<<std::endl);

    feature_use= split_model[1];
    outInfo("feature_use:"<<feature_use<<std::endl);

    return UIMA_ERR_NONE;
  }

  TyErrorId destroy()
  {
    outInfo("destroy");
    return UIMA_ERR_NONE;
  }


  TyErrorId processWithLock(CAS &tcas, ResultSpecification const &res_spec)
  {

    outInfo("RSSvmAnnotator is running:");
    rs::SceneCas cas(tcas);
    rs::Scene scene = cas.getScene();
    cas.get(VIEW_COLOR_IMAGE_HD, color);
    std::vector<rs::ObjectHypothesis> clusters;
    scene.identifiables.filter(clusters);


    if(feature_use == "VFH" || feature_use == "CVFH")
    {
      outInfo("Calculation starts with : " << set_mode << "::" << dataset_use << "::" << feature_use);
      svmObject->processPCLFeature(trained_model_name, set_mode, feature_use, clusters, svmObject, color, model_labels, tcas);
    }
    else if(feature_use == "CNN" || feature_use == "VGG16")
    {
      outInfo("Calculation starts with : " << set_mode << "::" << dataset_use << "::" << feature_use);
      svmObject->processCaffeFeature(trained_model_name, set_mode, feature_use, clusters, svmObject, color, model_labels, tcas);
    }
    else
    {
      outError("Please sellect the correct value of parameter(feature_use): VFH, CVFH, CNN, VGG16");
    }

    outInfo("calculation is done with SVM"<<std::endl);
    return UIMA_ERR_NONE;
  }

  void drawImageWithLock(cv::Mat &disp)
  {
    disp = color.clone();
  }

};

//This macro exports an entry point that is used to create the annotator.
MAKE_AE(SvmAnnotator)
