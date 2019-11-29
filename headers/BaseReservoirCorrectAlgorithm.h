#ifndef     BASE_RESERVOIR_CORRECT_ALGORITHM_H
#define     BASE_RESERVOIR_CORRECT_ALGORITHM_H

#include    "BasicLearnAlgorithm.h"

namespace nn {
    class BaseReservoirCorrectAlgorithm : public BasicLearnAlgorithm {
        
        public:

            BaseReservoirCorrectAlgorithm(){}
            virtual ~BaseReservoirCorrectAlgorithm(){}

            virtual void start(MultipleWeight& o_weights, Column& o_bias, const Column& i_totalIncomingSignal, BasicUnit *i_unit) override;
    };
}

#endif