#ifndef     RIDGE_REGRESSION_ALGORITHM_H
#define     RIDGE_REGRESSION_ALGORITHM_H


#include    "BasicLearnAlgorithm.h"

namespace nn {

    class RidgeRegressionAlgorithm : public BasicLearnAlgorithm {

        public:
            RidgeRegressionAlgorithm (const DataSet& i_learnSet, double i_ridge = 0.03);
            virtual ~RidgeRegressionAlgorithm(){}
            virtual void start(MultipleWeight& i_weight, Column& d_bias, const Column& d_activation, BasicUnit *i_unit) override;

        private:


            double  d_ridge;
    };
}

#endif