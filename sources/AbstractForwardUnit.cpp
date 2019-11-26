#include    "AbstractForwardUnit.h"

namespace nn {

    Data    AbstractForwardUnit::predict(const Data& i_input) {
        Data output = i_input;
        return output.each_col([this](const Column& v) {
            return this->operator()(v);
        });
    }

    Data    AbstractForwardUnit::predict2(const Column& i_initial, int i_predictHorizon) {
        ColumnContainer outContainer;
        Column  outColumn = i_initial;
        
        for (int step = 0 ; step < i_predictHorizon; ++step) {
            outColumn = this->operator()(outColumn);
            outContainer.push_back(outColumn);
        }

        return fn::formDataFromContainer(outContainer);
    }

    Column  AbstractForwardUnit::operator()(const Column& i_input) {
        return i_input;
    }

    bool    AbstractForwardUnit::learn(const Data& i_input, const Data& i_desired, int i_epochCount) {
        return true;
    }

    bool    AbstractForwardUnit::fit(const Data& i_input, int i_epochCount) {
        return true;
    }


    DoubleContainer AbstractForwardUnit::evaluate(const DataSet& i_learnSet, const DataSet& i_valideSet, const DataSet& i_testSet, int i_epochCount) {
        return {};
    }


}
