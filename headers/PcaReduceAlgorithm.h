#ifndef     PCA_REDUCE_ALGORITHM_H
#define     PCA_REDUCE_ALGORITHM_H


#include    "BasicLearnAlgorithm.h"


namespace nn {
    class PcaReducAlgorithm : public BasicLearnAlgorithm {

        public:

            explicit PcaReducAlgorithm(const MultipleData& i_inputData, int i_outDimention = -1);            
            virtual ~PcaReducAlgorithm(){}

            virtual void start(MultipleWeight& o_weights, Column& o_bias, Column& i_totalIncomingSignal, BasicUnit *i_unit) override;
        
        protected:

            const MultipleData& d_inputData;
            int d_outDimention;

    };
}


#endif