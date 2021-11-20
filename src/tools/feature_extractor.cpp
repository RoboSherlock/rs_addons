// Developed by: Rakib

#include <iostream>
#include <ros/ros.h>
#include <opencv2/opencv.hpp>


#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>


#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <vector>
#include <fstream>
#include <string>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <ros/package.h>

#ifdef WITH_CAFFE
#include <rs/recognition/CaffeProxy.h>
#endif

#include <dirent.h>
#include <yaml-cpp/yaml.h>
#include <pcl/io/pcd_io.h>
#include <algorithm>
#include <iterator>
#include <pcl/features/vfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/cvfh.h>
#include <pcl/features/our_cvfh.h>

using namespace cv;
using namespace std;
namespace po = boost::program_options;
namespace bfs = boost::filesystem;

//To read the split file for both the IAI dataset and BOTH datasets from rs_resource/object_datasets/splits folder
void readClassLabel(std::string obj_file_path,
                    std::vector <std::pair < string, double> > &objectToLabel,
                    std::vector <std::pair < string, double> > &objectToClassLabelMap)
{
  cv::FileStorage fs;
  fs.open(obj_file_path, cv::FileStorage::READ);
  std::vector<std::string> classes;

  cv::FileNode classesNode = fs["classes"];
  cv::FileNodeIterator it = classesNode.begin(), it_end = classesNode.end();
  for(; it != it_end; ++it) {
    classes.push_back(static_cast<std::string>(*it));
  }

  if(classes.empty()) {
    std::cout << "Object file has no classes defined" << std::endl;
  }
  else {
    double clslabel = 1;
    for(auto c : classes) {
      std::vector<std::string> subclasses;

      cv::FileNode subClassesNode = fs[c];
      cv::FileNodeIterator it = subClassesNode.begin(), it_end = subClassesNode.end();
      for(; it != it_end; ++it) {
        subclasses.push_back(static_cast<std::string>(*it));
      }

      //To set the map between string and double classlabel
      objectToClassLabelMap.push_back(std::pair< std::string, float >(c, clslabel));

      if(!subclasses.empty()) {
        for(auto sc : subclasses) {
          objectToLabel.push_back(std::pair< std::string, float >(sc, clslabel));
        }
      }
      else {
        objectToLabel.push_back(std::pair< std::string, float >(c, clslabel));
      }
      clslabel = clslabel + 1;
    }
  }
  fs.release();
  if(!objectToClassLabelMap.empty()) {
    std::cout << "objectToClassLabelMap:" << std::endl;
    for(unsigned int i = 0; i < objectToClassLabelMap.size(); i++)
      std::cout << objectToClassLabelMap[i].first << "::" << objectToClassLabelMap[i].second << std::endl;
  }
  std::cout << std::endl;

  if(!objectToLabel.empty()) {
    std::cout << "objectToLabel:" << std::endl;
    for(unsigned int i = 0; i < objectToLabel.size(); i++)
      std::cout << objectToLabel[i].first << "::" << objectToLabel[i].second << std::endl;

  }
  std::cout << std::endl;
}

//To read all the objects from rs_resources/objects_dataset folder.....................
void getFiles(const std::string &input_folder,
              std::vector <std::pair < string, double> > object_label,
              std::map<double, std::vector<std::string> > &modelFiles,
              std::string file_extension,
              const int skip_count=1)
{
  std::string path_to_data;
  if(boost::filesystem::exists(input_folder)) {
    path_to_data = input_folder;
  }
  else {
    std::string path_to_rs_resources = ros::package::getPath("rs_resources");
    path_to_data = path_to_rs_resources + "/objects_dataset/object_data/";
  }
  std::cerr << "______________________" << std::endl;
  std::cerr << path_to_data << std::endl;

  size_t pos;
  for(auto const &p : object_label) {
    std::string pathToObj("");
    std::cout << p.first;
    bfs::path bPath(path_to_data);

    try {
      bfs::recursive_directory_iterator dIt(bPath, bfs::symlink_option::recurse), end;
      while(dIt != end) {
        if(bfs::is_directory(dIt->path())) {

          if(dIt->path().filename().string() == p.first) {
            pathToObj = dIt->path().string();
            break;
          }
        }
        dIt++;
      }
    }
    catch(bfs::filesystem_error err) {
      std::cerr << err.what() << std::endl;
    }

    std::cout << "pathToObj:" << pathToObj << std::endl;
    if(pathToObj == "") {
      std::cout << "<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
      std::cout << "NO PATH FOUND TO FOLDER WITH THE NAME: " << p.first << std::endl;
      std::cout << "<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
      exit(1);
    }
    try {
      boost::filesystem::directory_iterator objDirIt(pathToObj);
      int idx =1;
      while(objDirIt != boost::filesystem::directory_iterator{}) {
        if(boost::filesystem::is_regular_file(objDirIt->path())) {
          std::string filename = objDirIt->path().string();
          pos = filename.rfind(file_extension.c_str());
          if(pos != std::string::npos) {
            if(idx%skip_count==0)
                modelFiles[p.second].push_back(objDirIt->path().string());
            idx++;
          }
        }
        objDirIt++;
      }
    }
    catch(bfs::filesystem_error err) {
      std::cerr << err.what() << std::endl;
    }
  }

  std::map<double, std::vector<std::string> >::iterator it;
  for(it = modelFiles.begin(); it != modelFiles.end(); ++it) {
    std::sort(it->second.begin(), it->second.end());
  }
}


void extractPCLDescriptors(std::string descriptorType,
                           const std::map<double, std::vector<std::string> > &modelFiles,
                           std::vector<std::pair<double, std::vector<float> > > &extract_features)
{
  std::string featDescription;
  for(std::map<double, std::vector<std::string> >::const_iterator it = modelFiles.begin();
      it != modelFiles.end(); ++it) {
    std::cerr << it->first << std::endl;
    for(uint32_t i = 0; i < it->second.size(); ++i) {
      std::cerr << it->second[i] << std::endl;
      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
      pcl::io::loadPCDFile(it->second[i], *cloud);

      pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne;
      ne.setInputCloud(cloud);

      pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBA> ());
      ne.setSearchMethod(tree);
      pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
      ne.setRadiusSearch(0.03);
      ne.compute(*cloud_normals);

      pcl::PointCloud<pcl::VFHSignature308>::Ptr extractedDiscriptor(new pcl::PointCloud<pcl::VFHSignature308> ());

      if(descriptorType == "VFH") {
        std::cout << "Calculation start with VFH Feature" << std::endl;

        pcl::VFHEstimation<pcl::PointXYZRGBA, pcl::Normal, pcl::VFHSignature308> vfhEstimation;
        vfhEstimation.setInputCloud(cloud);
        vfhEstimation.setInputNormals(cloud_normals);
        vfhEstimation.setNormalizeBins(true);
        vfhEstimation.setNormalizeDistance(true);
        vfhEstimation.setSearchMethod(tree);
        vfhEstimation.compute(*extractedDiscriptor);

        featDescription = "VFH feature size :";
      }

      if(descriptorType == "CVFH") {
        std::cout << "Calculation start with CVFH Feature" << std::endl;
        pcl::CVFHEstimation<pcl::PointXYZRGBA, pcl::Normal, pcl::VFHSignature308> cvfhEst;
        cvfhEst.setInputCloud(cloud);
        cvfhEst.setInputNormals(cloud_normals);
        cvfhEst.setSearchMethod(tree);
        cvfhEst.setEPSAngleThreshold(5.0 / 180.0 * M_PI); // 5 degrees.
        cvfhEst.setCurvatureThreshold(1.0);
        cvfhEst.setNormalizeBins(true);
        cvfhEst.compute(*extractedDiscriptor);
        featDescription = "CVFH feature size :";
      }

      std::vector<float> descriptorVec;
      descriptorVec.resize(308);
      for(size_t j = 0; j < 308; ++j) {
        descriptorVec[j] = extractedDiscriptor->points[0].histogram[j];
      }

      cv::Mat desc(1, descriptorVec.size(), CV_32F, &descriptorVec[0]);
      cv::normalize(desc, desc, 1, 0, cv::NORM_L2);
      std::vector<float> descNormed;
      descNormed.assign((float *)desc.datastart, (float *)desc.dataend);

      extract_features.push_back(std::pair<double, std::vector<float> >(it->first, descNormed));
    }
  }
  std::cerr << featDescription << extract_features.size() << std::endl;
}

#ifdef WITH_CAFFE
void extractCaffeFeature(std::string featType,
                         const  std::map<double, std::vector<std::string> > &modelFiles,
                         std::string resourcesPackagePath,
                         std::vector<std::pair<double, std::vector<float> > > &caffe_features)
{
  std::string CAFFE_MODEL_FILE;
  std::string CAFFE_TRAINED_FILE;
  std::string featDescription;

  if(featType == "VGG16") {
    CAFFE_MODEL_FILE = "/caffe/models/bvlc_reference_caffenet/VGG_ILSVRC_16_layers_deploy.prototxt";
    CAFFE_TRAINED_FILE = "/caffe/models/bvlc_reference_caffenet/VGG_ILSVRC_16_layers.caffemodel";
    featDescription = "VGG16 feature size :";
  }
  else if(featType == "BVLC_REF") {
    CAFFE_MODEL_FILE = "/caffe/models/bvlc_reference_caffenet/deploy.prototxt";
    CAFFE_TRAINED_FILE = "/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel";
    featDescription = "BVLC_REF feature size :";
  }
  else {
    std::cerr << "CAFFE_MODEL_FILE and CAFFE_TRAINED_FILE are not found" << std::endl;
    exit(0);
  }

  std::string CAFFE_MEAN_FILE = "/caffe/data/imagenet_mean.binaryproto";
  std::string CAFFE_LABLE_FILE = "/caffe/data/synset_words.txt";

  CaffeProxy caffeProxyObj(resourcesPackagePath + CAFFE_MODEL_FILE,
                           resourcesPackagePath + CAFFE_TRAINED_FILE,
                           resourcesPackagePath + CAFFE_MEAN_FILE,
                           resourcesPackagePath + CAFFE_LABLE_FILE);

  for(std::map<double, std::vector<std::string>>::const_iterator it = modelFiles.begin();
      it != modelFiles.end(); ++it) {
    std::cerr << it->first << std::endl;
    for(int i = 0; i < it->second.size(); ++i) {
      std::cerr << it->second[i] << std::endl;
      cv::Mat rgb = cv::imread(it->second[i]);
      std::vector<float> feature = caffeProxyObj.extractFeature(rgb);

      cv::Mat desc(1, feature.size(), CV_32F, &feature[0]);
      cv::normalize(desc, desc, 1, 0, cv::NORM_L2);
      std::vector<float> descNormed;
      descNormed.assign((float *)desc.datastart, (float *)desc.dataend);
      caffe_features.push_back(std::pair<double, std::vector<float>>(it->first, descNormed));
    }
  }
  std::cerr << featDescription << caffe_features.size() << std::endl;
}
#endif

// To split the instance dataset into train and and test dataset..............................
void splitDataset(std::vector<std::pair<double, std::vector<float> > > features,
                  std::vector<std::pair<double, std::vector<float> > > &output_train,
                  std::vector<std::pair<double, std::vector<float> > > &output_test)
{
  //every fourth descriptor stored to vector output_test
  for(uint32_t i = 0; i < features.size() / 4; i++) {
    for(uint8_t j = 0; j < 3; j++) {
      output_train.push_back(features[j + 4 * i]);
    }
    output_test.push_back(features[3 + 4 * i]);
  }
}

// To split the instance dataset into train and and test dataset..............................
void descriptorsSplit(std::vector<std::pair<double, std::vector<float> > > features,
                      std::vector<std::pair<double, std::vector<float> > > &output)
{
  // The following loop split every fourth descriptor and store it to vector output_test
  for(uint32_t i = 0; i < features.size() / 4; i++) {
    output.push_back(features[4 * i]);
  }
}

// To save the train and test data in cv::Mat format in folder /rs_resource/extractedFeat
void saveDatasets(std::vector<std::pair<double, std::vector<float> > > data,
                  std::string descriptor_name,
                  std::string split_name, std::string savePathToOutput)
{
  if(data.size() == 0) return;
  cv::Mat descriptors_train(data.size(), data[0].second.size(), CV_32F);
  cv::Mat label_train(data.size(), 1, CV_32F);


  for(size_t i = 0; i <  data.size(); ++i) {
    label_train.at<float>(i, 0) = static_cast<float>(data[i].first);

    for(size_t j = 0; j <  data[i].second.size(); ++j) {
      descriptors_train.at<float>(i, j) =  data[i].second[j];
    }
  }
  //To save file in disk...........................................................
  cv::FileStorage fs;
  // To save the train data.................................................
  fs.open(savePathToOutput + "/" + descriptor_name + '_' + "data_"+split_name +".yaml", cv::FileStorage::WRITE);
  fs << "label" << label_train;
  fs << "descriptors" << descriptors_train;
  fs.release();

  std::cout << "extracted feautres should be found in path (" << savePathToOutput << ")" << std::endl;
}

void saveObjectToLabels(std::vector <std::pair < string, double> > input_file, std::string descriptor_name,
                        std::string split_name, std::string savePathToOutput)
{
  std::ofstream file((savePathToOutput + "/" +descriptor_name + '_' + "ClassLabel_"+split_name+ ".txt").c_str());
  for(auto p : input_file) {
    file << p.first << ":" << p.second << endl;
  }
  std::cout << " clasLabel in double should be found in path (" << savePathToOutput << ")" << std::endl;
}

int main(int argc, char **argv)
{
  po::options_description desc("Allowed options");
  std::string split_file, feat, input_folder, output_folder, split_name;
  int leave_count;
  desc.add_options()
  ("help,h", "Print help messages")
  ("split,s", po::value<std::string>(&split_file)->default_value("breakfast3"),
   "enter the split file name")
  ("feature,f", po::value<std::string>(&feat)->default_value("BVLC_REF"),
   "choose feature to extract: [BVLC_REF|VGG16|VFH|CVFH]")
  ("input,i", po::value<std::string>(&input_folder)->default_value(""),
   "set input location for image data")
  ("output,o", po::value<std::string>(&output_folder)->default_value(""),
   "set output location")
  ("leave_out,l", po::value<int>(&leave_count)->default_value(1), "if x>1 take only every x_th file for extractions");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if(vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }



  // Define path to get the datasets.......................................................
  std::string resourcePath = ros::package::getPath("rs_resources");
  std::string split_file_path ;
  if(bfs::exists(bfs::path(split_file))) {
    split_file_path = split_file;
  }
  else {
    split_file_path = resourcePath + "/objects_dataset/splits/" + split_file;
  }

  if(!bfs::exists(bfs::path(split_file_path))) {
    std::cout << "*********************************************************************************************" << std::endl;
    std::cerr << " Split file (.yaml) is not found. Please check the path below  :" << std::endl;
    std::cerr << "Path to class label file : " << split_file_path << std::endl << std::endl;
    std::cerr << "The file should be in ( rs_resources/objects_datasets/splits/ ) folder " << std::endl << std::endl;
    return EXIT_FAILURE;
  }

  bfs::path p(split_file_path);
  split_name = p.stem().string();
  std::cout << "Path to split file : " << split_file_path << std::endl;

  //To save file in disk...........................................................
  std::string savePathToOutput;
  if(bfs::exists(output_folder)) {
    savePathToOutput = output_folder;
  }
  else {
    savePathToOutput = resourcePath + "/extracted_feats/";
  }

  // To check the storage folder for generated files by this program ................................................
  if(!bfs::exists(savePathToOutput)) {
    //   boost::filesystem::create_directory(savePathToOutput);

    std::cerr << savePathToOutput << "output folder does not exist!" << std::endl;
    exit(1);
  }

  std::cout << "Path to save the extracted feature : " << savePathToOutput << std::endl << std::endl;

  std::vector <std::pair < string, double> > objectToLabel;
  std::vector <std::pair < string, double> > objectToClassLabelMap;

  // To read the class label from .yaml file................

  readClassLabel(split_file_path, objectToLabel, objectToClassLabelMap);


  // need to store .pcd  or .png file from storage
  std::map< double, std::vector<std::string> > model_files_all;

  //Extract the feat descriptors
  std::vector<std::pair<double, std::vector<float> > > descriptors_all;

  if(feat == "BVLC_REF" || feat == "VGG16") {
#ifdef WITH_CAFFE
    std::cout << "Calculation starts with :" << "::" << feat << std::endl;
    // To read all .png files from the storage folder...........
    getFiles(input_folder, objectToLabel, model_files_all, "_crop.png", leave_count);
    extractCaffeFeature(feat, model_files_all, resourcePath, descriptors_all);
#else
    std::cerr << "Caffe not available." << std::endl;
    exit(1);
#endif
  }
  else if(feat == "VFH" || feat == "CVFH") {
    std::cout << "Calculation starts with :" << "::" << feat << std::endl;
    // To read all .cpd files from the storage folder...........
    getFiles(input_folder, objectToLabel, model_files_all, ".pcd", leave_count);
    // To calculate VFH descriptors..................................
    extractPCLDescriptors(feat, model_files_all, descriptors_all);
  }
  else {
    std::cerr << "Please select one of the supported feature descriptors (BVLC_REF, VGG16, VFH, CVFH)" << std::endl;
    return EXIT_FAILURE;
  }

  // To split all the calculated VFH descriptors (descriptors_all) into train and test data for
  // the classifier. Here evey fourth element of vector (descriptors_all) is considered as test data
  // and rest are train data
  //splitDataset(descriptors_all, descriptors_all_train, descriptors_all_test);
  // To save the train and test data in path /rs_resources/objects_dataset/extractedFeat
  saveDatasets(descriptors_all, feat, split_name, savePathToOutput);

  //To save the string class labels in type double in folder rs_resources/objects_dataset/extractedFeat
  saveObjectToLabels(objectToClassLabelMap, feat, split_name, savePathToOutput);

  std::cout << "Descriptors calculation is done" << std::endl;
  return 0;
}
