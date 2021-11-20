// Developed by: Rakib

#include <uima/api.hpp>
#include <robosherlock/utils/time.h>
#include <rs_addons/classifiers/RSSVM.h>
#include <rs_addons/classifiers/RSRF.h>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

using namespace uima;
namespace bpo = boost::program_options;

int main(int argc, char **argv)
{
    bpo::options_description desc("program options");

    //classifier type should be rssvm, rsrf, rsgbt, rsknn........
    std::string classifier_type;
    //the name of train matrix file in folder /rs_resources/objects_dataset/extractedFeat
    std::string train_data_name;
    //the name of train label matrix file in folder /rs_resources/objects_dataset/extractedFeat
    std::string train_label_name;
    // the name of trained model file, which will be generated in folder rs_addons/trainedData
    std::string trained_model_name;

    desc.add_options()
            ("help,h", "Print help messages")
            ("classifier,c", bpo::value<std::string>(&classifier_type)->default_value("SVM"),
             "classifier to train: [SVM|RF|KNN]")
            ("input,i", bpo::value<std::string>(&train_data_name)->default_value(""),
             "enter input features file")
            ("out,o", bpo::value<std::string>(&trained_model_name)->default_value("classifier.model"),
             "output file name");
    bpo::variables_map vm;
    bpo::store(bpo::parse_command_line(argc, argv, desc), vm);
    bpo::notify(vm);

    if(vm.count("help")) {
      std::cout << desc << "\n";
      return 1;
    }

    outInfo("Name of the loaded files for classifier trainning are:"<<std::endl);
    outInfo("classifier_type:"<<classifier_type<<std::endl);
    outInfo("train_data_name:"<<train_data_name<<std::endl);
    outInfo("train_label_name:"<<train_label_name<<std::endl);

    //vector to hold split trained_model_name
    std::vector<std::string> split;
    boost::split(split, train_data_name, boost::is_any_of("_"));

    trained_model_name= split[0]+'_'+split[1]+'_'+classifier_type+"Model"+'_'+split[3];
    outInfo("trained_model_name:"<<trained_model_name<< "  will be generated in rs_addons/trainedData");

    if(classifier_type=="SVM")
    {
        RSClassifier* svmObject= new RSSVM;
        outInfo("Training with SVM is going on .......");
        svmObject->trainModel(train_data_name ,train_label_name, trained_model_name);
    }
    else if(classifier_type=="RF")
    {
        RSClassifier* rfObject= new RSRF;
        outInfo("Training with RSRF is going on .......");
        rfObject->trainModel(train_data_name ,train_label_name, trained_model_name);
    }
    else if(classifier_type=="KNN")
    {
        outInfo("No need to train a KNN;");
    }
    else
    {
        outError("Please select the correct classifier_type, which is either rssvm, rsrf, rsgbt, rsknn");
    }
    outInfo("Classifier training is Done !!!");
}

