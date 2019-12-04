#include    "BasicUnit.h"
#include    "BasicLearnAlgorithm.h"

namespace nn {
    BasicUnit::BasicUnit(int i_neuronsAmount, Activation i_func) 
        : d_neuronsAmount(i_neuronsAmount), d_activFunc(i_func)
    {
        d_bias = MathVector(Column, randn, d_neuronsAmount);
        d_totalIncomingSignal = MathVector(Column, zeros, d_neuronsAmount);
    }

    BasicUnit::~BasicUnit() {}

    bool BasicUnit::init(const MultipleVector& i_initialInput) {
        if (i_initialInput.empty()) {
            ERROR_LOG("BaseUnit::init -> initial list is empty");
            return false;
        }

        for (const auto& v: i_initialInput)
            if (v.n_elem > 0)
                d_inputWeights.push_back(
                    RandnMatrix(d_neuronsAmount, v.n_elem));
            else {
                ERROR_LOG("BaseUnit::init -> one element in initial list is empty");
                return false;
            }
        return true;
    }

    Column  BasicUnit::forward(const MultipleVector& i_inputSignales) {
        try {
            totalIncomingSignal(i_inputSignales);
            return out();
        } catch (std::exception& e) {
            THROW_FORWARD("BaseUnit::forward -> ", e);
        }

        return {};
    }

    const Column&  BasicUnit::totalIncomingSignal(const MultipleVector& i_inputSignales) {
        d_totalIncomingSignal = MathVector(Column, zeros, d_bias.n_elem);
        try {
            auto weightIter = d_inputWeights.begin();
            auto inputIter = i_inputSignales.begin();
            for (; weightIter != d_inputWeights.end() && inputIter != i_inputSignales.end();
                    ++weightIter, ++inputIter)
            {
                d_totalIncomingSignal += (*weightIter)*(*inputIter);
            }    
            if (weightIter != d_inputWeights.end() || inputIter != i_inputSignales.end())
                throw std::runtime_error ("sizes of inputs and weights list are different");
        } catch (std::exception& e) {
            THROW_FORWARD("totalIncomingSignal -> ", e);
        }
        return d_totalIncomingSignal;
    }

    Column  BasicUnit::out() {
        Column result = d_totalIncomingSignal + d_bias;
        return d_activFunc ? result.transform(d_activFunc) : result;
    }

    bool BasicUnit::learn(BasicLearnAlgorithm* i_algorithm) {
        try {
            if (!i_algorithm) throw std::runtime_error("there is no algorithm");
            i_algorithm->start(d_inputWeights, d_bias, d_totalIncomingSignal, this);
        } catch (std::exception& e) {
            THROW_FORWARD("BasicUnit::learn -> ", e);
            return false;
        }
        return true;
    }

}