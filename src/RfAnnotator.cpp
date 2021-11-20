// Developed by: Rakib

#include <uima/api.hpp>
#include <iostream>
#include <robosherlock/types/all_types.h>
#include <robosherlock/scene_cas.h>
#include <robosherlock/utils/time.h>
#include <opencv2/core.hpp>
#include<ros/package.h>
#include <rs_addons/classifiers/RSRF.h>
#include <robosherlock/DrawingAnnotator.h>

using namespace uima;

class RfAnnotator : public DrawingAnnotator
{
private:

  cv::Mat color;

  //set_mode should be GT(groundTruth) or CF (classify) ...........
  std::string set_mode;

  //dataset_use should be IAI (kitchen data from IAI) or WU (data from Washington University) or both...
  std::string dataset_use;

  //feature_use should be VFH, CVFH, CNN, VGG16 .....
  std::string feature_use;

  //the name of trained model ifle in folder rs_addons/trainedData/
  std::string trained_model_name;

  //vector to hold split trained_model_name
  std::vector<std::string> split_model;

  //the name of actual_class_label map file in path rs_resources/objects_dataset/extractedFeat/
  std::string actual_class_label;

  //vector to hold classes name
  std::vector<std::string> model_labels;

public:

  RfAnnotator(): DrawingAnnotator(__func__)
  {}

  RSClassifier* rfObject= new RSRF;

  TyErrorId initialize(AnnotatorContext &ctx)
  {
    outInfo("initialize");
    ctx.extractValue("set_mode", set_mode);
    ctx.extractValue("trained_model_name", trained_model_name);
    ctx.extractValue("actual_class_label", actual_class_label);

    outInfo("Name of the loaded files for RF are:"<<std::endl);

    outInfo("set_mode:"<<set_mode<<std::endl);
    outInfo("trained_model_name:"<<trained_model_name<<std::endl);
    outInfo("actual_class_label:"<<actual_class_label<<std::endl);

    rfObject->setLabels(actual_class_label, model_labels);

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
    outInfo("RSRFAnnotator is running:");
    rs::SceneCas cas(tcas);
    rs::Scene scene = cas.getScene();
    cas.get(VIEW_COLOR_IMAGE_HD, color);
    std::vector<rs::ObjectHypothesis> clusters;
    scene.identifiables.filter(clusters);


    if(feature_use == "VFH" || feature_use == "CVFH")
    {
      outInfo("Calculation starts with : " << set_mode << "::" << dataset_use << "::" << feature_use);
      rfObject->processPCLFeature(trained_model_name,set_mode,feature_use,clusters, rfObject, color,model_labels, tcas);
    }
    else if(feature_use == "CNN" || feature_use == "VGG16")
    {
      outInfo("Calculation starts with : " << set_mode << "::" << dataset_use << "::" << feature_use);
      rfObject->processCaffeFeature(trained_model_name,set_mode,feature_use,clusters, rfObject, color, model_labels, tcas);
    }
    else
    {
      outError("Please sellect the correct value of parameter(feature_use): VFH, CVFH, CNN, VGG16");
    }

    outInfo("calculation is done with RSRF"<<std::endl);
    return UIMA_ERR_NONE;
  }

  void drawImageWithLock(cv::Mat &disp)
  {
    disp = color.clone();
  }

};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(RfAnnotator)
