#ifndef     INFO_MAX_ONE_ALGORITHM_H
#define     INFO_MAX_ONE_ALGORITHM_H


#include    "BasicLearnAlgorithm.h"

namespace nn {
    class InfoMaxOneAlgorithm : public BasicLearnAlgorithm {
        public:

            InfoMaxOneAlgorithm(const MultipleData& i_data, const int i_iterations = 10, double i_learnSpead = 1e-10);
            virtual~InfoMaxOneAlgorithm();

            virtual void start(MultipleWeight& io_weights, Column& io_bias, Column& i_totalIncomingSignal, BasicUnit *i_unit) override;

        protected:

            Column unitOut(BasicUnit* i_unit, const MultipleVector& i_inp, Column& io_bias, Column& io_totalIncomingSignal);
            const Column& deltaOptBias(const Column& i_unitInp, const Column& i_unitOut);
            const Column& deltaTisGain();

        protected:

            int d_iterations;
            const MultipleData& d_data;

            Column  d_tisGain;
            Column  d_optBias;

            Column  d_deltaTisGain;
            Column  d_deltaOptBias;

            double  d_learnSpead;
    };
}


#endif