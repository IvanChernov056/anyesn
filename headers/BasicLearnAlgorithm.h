#ifndef     BASIC_LEARN_ALGORITHM_H
#define     BASIC_LEARN_ALGORITHM_H

#include    "BasicUnit.h"

namespace nn {
    class   BasicLearnAlgorithm {
        public:
            BasicLearnAlgorithm (const DataSet& i_learnSet);
            virtual ~BasicLearnAlgorithm(){}
            virtual void start(MultipleWeight& i_weight, Column& d_bias, const Column& d_activation, BasicUnit *i_unit);
        
        protected:

            const DataSet& d_dataSet;
    };
}

#endif