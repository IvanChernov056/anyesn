#ifndef     SIMPLE_UNIT_H
#define     SIMPLE_UNIT_H

#include    "AbstractForwardUnit.h"


namespace nn {
    class SimpleUnit : public AbstractForwardUnit {

        public:
            SimpleUnit(int i_neuronsNumber = 1, bool i_useBias = false, Activation i_func = nullptr);
            virtual ~SimpleUnit(){}

            virtual Column  operator()(const Column& i_input) override;
            virtual bool    fit(const Data& i_input, int i_epochCount) override;

        protected:

            Matrix  d_weight;
            Column  d_bias;
            Activation  d_activFunc;

            bool    d_useBias;
    };
}

#endif