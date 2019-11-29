#ifndef     BASIC_LEARN_ALGORITHM_H
#define     BASIC_LEARN_ALGORITHM_H

#include    "BasicUnit.h"

namespace nn {
    class   BasicLearnAlgorithm {
        public:
            BasicLearnAlgorithm (){}
            virtual ~BasicLearnAlgorithm(){}
            virtual void start(MultipleWeight& o_weights, Column& o_bias, const Column& i_totalIncomingSignal, BasicUnit *i_unit);
    };
}

#endif