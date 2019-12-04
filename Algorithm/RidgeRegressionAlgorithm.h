#ifndef     RIDGE_REGRESSION_ALGORITHM_H
#define     RIDGE_REGRESSION_ALGORITHM_H


#include    "BasicLearnAlgorithm.h"

namespace nn {

    class RidgeRegressionAlgorithm : public BasicLearnAlgorithm {

        public:
            RidgeRegressionAlgorithm (const MultipleDataSet& i_learnSet, double i_ridge = 0.03);
            virtual ~RidgeRegressionAlgorithm(){}
            virtual void start(MultipleWeight& o_weights, Column& o_bias,  Column& i_totalIncomingSignal, BasicUnit *i_unit) override;

        private:

            const MultipleDataSet& d_dataSet;
            double  d_ridge;
    };
}

#endif