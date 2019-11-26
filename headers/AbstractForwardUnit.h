#ifndef     ABSTRACT_FORWARD_UNIT_H
#define     ABSTRACT_FORWARD_UNIT_H

#include    "nnutils.h"

namespace   nn {

    class   AbstractForwardUnit {

        public:

            virtual Data    predict(const Data& i_input);
            virtual Data    predict2(const Column& i_initial, int i_predictHorizon);
            virtual Column  operator()(const Column& i_input);
            virtual bool    learn(const Data& i_input, const Data& i_desired, int i_epochCount);
            virtual bool    fit(const Data& i_input, int i_epochCount);

            virtual DoubleContainer evaluate(const DataSet& i_learnSet, const DataSet& i_valideSet, const DataSet& i_testSet, int i_epochCount);
    };
}

#endif