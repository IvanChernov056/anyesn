#include    "BasicLearnAlgorithm.h"
#include    "BasicUnit.h"


namespace nn {
    BasicLearnAlgorithm::BasicLearnAlgorithm (const DataSet& i_learnSet) : d_dataSet(i_learnSet) {
        
    }

    void BasicLearnAlgorithm::start(MultipleWeight& i_weights, Column& d_bias, const Column& d_activation, BasicUnit *i_unit){
        INFO_LOG("\nbefore forward :\n" << d_activation);
        Column a = i_unit->forward(d_dataSet.first[0]);
        INFO_LOG("\nafter forward :\n" << d_activation);
    }

}