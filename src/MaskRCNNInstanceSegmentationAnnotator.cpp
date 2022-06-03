#include <vector>

// python
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

// opencv
#include <opencv2/opencv.hpp>

#include <pcl/point_types.h>
#include <rs/types/all_types.h>
//RS
#include <rs/scene_cas.h>
#include <rs/DrawingAnnotator.h>
#include <rs/utils/common.h>
#include <rs/utils/time.h>


namespace python = boost::python;
namespace np = boost::numpy;

using namespace uima;


class MaskRCNNInstanceSegmentationAnnotator : public DrawingAnnotator
{
private:
  struct ObjectMask
  {
    int label;
    cv::Rect bbox;
    cv::Mat mask;
    float score;
  };

  std::vector<ObjectMask> masks;
  cv::Mat color;

  std::string structure;
  std::string pretrained_model;
  int gpu;
  float score_thresh;
  python::object predictor;
  python::tuple label_names;

public:

  MaskRCNNInstanceSegmentationAnnotator(): DrawingAnnotator(__func__) {}

  TyErrorId initialize(AnnotatorContext &ctx)
  {
    outInfo("initialize");
    ctx.extractValue("structure", structure);
    ctx.extractValue("pretrained_model", pretrained_model);
    ctx.extractValue("gpu", gpu);
    ctx.extractValue("score_thresh", score_thresh);

    Py_Initialize();
    np::initialize();
    python::object rs_test_module = python::import("rs_addons");
    predictor = rs_test_module.attr("MaskRCNNInstanceSegmentationPredictor")
        (structure.c_str(), pretrained_model.c_str(), gpu, score_thresh);
    label_names = python::extract<python::tuple>(predictor.attr("label_names"));
    return UIMA_ERR_NONE;
  }

  TyErrorId destroy()
  {
    outInfo("destroy");
    return UIMA_ERR_NONE;
  }

private:
  TyErrorId processWithLock(CAS &tcas, ResultSpecification const &res_spec)
  {
    outInfo("process start");
    rs::StopWatch clock;
    rs::SceneCas cas(tcas);
    rs::Scene scene = cas.getScene();

    cas.get(VIEW_COLOR_IMAGE_HD, color);
    cv::Size color_size = color.size();
    int color_height = color_size.height;
    int color_width = color_size.width;

    python::tuple shape = python::make_tuple(color_height, color_width, 3);
    np::ndarray img = np::zeros(shape, np::dtype::get_builtin<uint8_t>());
    uint8_t* img_ptr = reinterpret_cast<uint8_t *>(img.get_data());
    for (int i=0; i < color_width * color_height * 3; i++)
    {
      *(img_ptr + i) = color.data[i];
    }

    outInfo("MaskRCNN prediction start");
    python::tuple return_tuple = python::extract<python::tuple>(
      predictor.attr("predict")(img));
    outInfo("MaskRCNN prediction finish");

    python::list mask_list = python::extract<python::list>(return_tuple[0]);
    np::ndarray bbox = python::extract<np::ndarray>(return_tuple[1]);
    np::ndarray label = python::extract<np::ndarray>(return_tuple[2]);
    np::ndarray score = python::extract<np::ndarray>(return_tuple[3]);
    int n_rois = bbox.get_shape()[0];
    masks.resize(n_rois);

    int *bbox_ptr = reinterpret_cast<int *>(bbox.get_data());
    int *label_ptr = reinterpret_cast<int *>(label.get_data());
    float *score_ptr = reinterpret_cast<float *>(score.get_data());

    for (int i = 0; i < n_rois; i++) {
      // bbox
      int y_min = std::max(0, *(bbox_ptr + 4 * i));
      int x_min = std::max(0, *(bbox_ptr + 4 * i + 1));
      int y_max = std::min(color_height, *(bbox_ptr + 4 * i + 2));
      int x_max = std::min(color_width, *(bbox_ptr + 4 * i + 3));

      cv::Rect bbox_hires, bbox_lowres;
      bbox_lowres = cv::Rect(
        x_min / 2, y_min / 2, (x_max - x_min) / 2, (y_max - y_min) / 2);
      bbox_hires = cv::Rect(x_min, y_min, x_max - x_min, y_max - y_min);

      // mask
      cv::Mat mask_hires, mask_lowres;
      mask_hires = cv::Mat::zeros(bbox_hires.height, bbox_hires.width, CV_8U);
      np::ndarray mask = python::extract<np::ndarray>(mask_list[i]);
      uint8_t *mask_ptr = reinterpret_cast<uint8_t *>(mask.get_data());
      int stride = mask.strides(0);
      for (int y = 0; y < bbox_hires.height; ++y) {
          for (int x = 0; x < bbox_hires.width; ++x) {
              mask_hires.at<uint8_t>(y, x) = mask_ptr[x + y * stride];
          }
      }

      cv::resize(mask_hires, mask_lowres, bbox_lowres.size(), 0, 0, cv::INTER_NEAREST);

      masks[i].mask = mask_hires;
      masks[i].bbox = bbox_hires;
      masks[i].label = *(label_ptr + i);
      masks[i].score = *(score_ptr + i);

      std::string label_name = python::extract<std::string>(label_names[masks[i].label]);
      outInfo("label " << i << "th: " << label_name);
      outInfo("score " << i << "th: " << masks[i].score);
      outInfo("bbox " << i << "th: ("
        << y_min << ", " << x_min << ", " << y_max << ", " << x_max << ")");

      // annotate
      rs::ImageROI roi = rs::create<rs::ImageROI>(tcas);
      roi.roi(rs::conversion::to(tcas, bbox_lowres));
      roi.roi_hires(rs::conversion::to(tcas, bbox_hires));
      roi.mask(rs::conversion::to(tcas, mask_lowres));
      roi.mask_hires(rs::conversion::to(tcas, mask_hires));
      rs::Cluster cluster = rs::create<rs::Cluster>(tcas);
      cluster.rois(roi);
      rs::Classification classification = rs::create<rs::Classification>(tcas);
      classification.classification_type.set("CLASS");
      classification.classname.set(label_name);
      classification.classifier.set("MaskRCNN");
      classification.source.set("MaskRCNNInstanceSegmentationAnnotator");
      cluster.annotations.append(classification);
      scene.identifiables.append(cluster);
    }

    return UIMA_ERR_NONE;
  }

  void drawImageWithLock(cv::Mat &disp)
  {
    disp = color.clone();
    for(int i = 0; i < masks.size(); i++)
    {
      int label = masks[i].label;
      float score = masks[i].score;
      std::string label_name = python::extract<std::string>(label_names[label]);
      std::ostringstream ss;
      ss << "label: " << label_name << ", score: " << score;
      std::string bbox_text = ss.str();
      cv::Rect bbox = masks[i].bbox;
      cv::Scalar color = rs::common::cvScalarColors[i % rs::common::numberOfColors];
      cv::putText(disp, bbox_text, cv::Point(bbox.x, bbox.y), 0, 0.5, color);
      cv::rectangle(disp, bbox, color);

      cv::Mat &mask = masks[i].mask;
      for (int j = 0; j < bbox.height; j++) {
        int y = j + bbox.y;
        for (int k = 0; k < bbox.width; k++) {
          int x = k + bbox.x;
          if (mask.at<uint8_t>(j, k) > 0) {
            for (int c = 0; c < disp.channels(); c++) {
              disp.at<cv::Vec3b>(y, x) = cv::Vec3b(color[0], color[1], color[2]);
            }
          }
        }
      }
    }
  }
};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(MaskRCNNInstanceSegmentationAnnotator)
