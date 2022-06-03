// Developed by: Rakib

#include <uima/api.hpp>
#include <iostream>
#include <pcl/point_types.h>


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>


#include <ros/package.h>

#include <robosherlock/types/all_types.h>
#include <robosherlock/scene_cas.h>
#include <robosherlock/utils/time.h>
#include <robosherlock/DrawingAnnotator.h>

#include <rs_addons/classifiers/RSKNN.h>

using namespace uima;

class KnnAnnotator : public DrawingAnnotator
{
private:

    cv::Mat color;

    //set_mode should be GT(groundTruth) or CL (classify)....
    std::string mode;

    //the value of k-neighbors in the knn-classifier
    int default_k;

    //feature_use should be VFH, CVFH, CNN, VGG16 .....
    std::string feature_use;

    //the name of train matrix and its labels in path rs_resources/objects_dataset/extractedFeat/
    std::string trainKNN_matrix;
    std::string trainKNNLabel_matrix;

    //vector to hold split trained_model_name
    std::vector<std::string> split_model;

    //the name of actual_class_label map file in path rs_resources/objects_dataset/extractedFeat/
    std::string classNrMapping;

    //vector to hold classes name
    std::vector<std::string> model_labels;

    RSKNN *knnObject;

public:

    KnnAnnotator(): DrawingAnnotator(__func__)
    {}



    TyErrorId initialize(AnnotatorContext &ctx)
    {
        ctx.extractValue("set_mode", mode);
        outInfo("set_mode:" << mode << std::endl);

        ctx.extractValue("default_k", default_k);
        outInfo("Value of k-neighbors: " << default_k);

        ctx.extractValue("training_data", trainKNN_matrix);
        outInfo("training_data:" << trainKNN_matrix);

        ctx.extractValue("class_label_mapping", classNrMapping);
        outInfo("class_label_mapping:" << classNrMapping );



        ctx.extractValue("feature_descriptor_type", feature_use);
        outInfo("feature descriptor set: "<<feature_use);

        knnObject = new RSKNN(default_k);
        knnObject->loadModelFile(trainKNN_matrix);
        knnObject->setLabels(classNrMapping, model_labels);

        return UIMA_ERR_NONE;
    }

    TyErrorId destroy()
    {
        outInfo("destroy");
        return UIMA_ERR_NONE;
    }

    TyErrorId processWithLock(CAS &tcas, ResultSpecification const &res_spec)
    {
        rs::StopWatch clock;
        outInfo("RSKNNAnnotator is running:");
        rs::SceneCas cas(tcas);
        rs::Scene scene = cas.getScene();
        cas.get(VIEW_COLOR_IMAGE_HD, color);
        std::vector<rs::ObjectHypothesis> clusters;
        scene.identifiables.filter(clusters);
        outInfo("Feature to use: "<<feature_use);
        if(feature_use == "VFH" || feature_use == "CVFH") {
            outInfo("Calculation starts with : " << mode  << "::" << feature_use);
            knnObject->processPCLFeatureKNN(mode, feature_use, clusters, color, model_labels, tcas);
        }
        else if(feature_use == "BVLC_REF" || feature_use == "VGG16") {
            outInfo("Calculation starts with : " << mode << "::" << feature_use);
            knnObject->processCaffeFeatureKNN(mode, feature_use, clusters, color, model_labels, tcas);
        }
        else {
            outError("Please sellect the correct value of parameter(feature_use): VFH, CVFH, BVLC_REF, VGG16");
        }

        outInfo("calculation is done with RSKNN" << std::endl);
        outInfo("took: " << clock.getTime() << " ms. with the Cluster size of : " << clusters.size());
        return UIMA_ERR_NONE;
    }

    void drawImageWithLock(cv::Mat &disp)
    {
        disp = color.clone();
    }

};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(KnnAnnotator)
