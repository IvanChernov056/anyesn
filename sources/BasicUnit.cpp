#include    "BasicUnit.h"
#include    "BasicLearnAlgorithm.h"

namespace nn {
    BasicUnit::BasicUnit(int i_neuronsAmount, Activation i_func) 
        : d_neuronsAmount(i_neuronsAmount), d_activFunc(i_func)
    {}
    BasicUnit::~BasicUnit() {}

    bool BasicUnit::init(const MultipleVector& i_initialInput) {
        if (i_initialInput.empty()) {
            ERROR_LOG("BaseUnit::init : initial list is empty");
            return false;
        }
        
        d_bias = MathVector(Column, randn, d_neuronsAmount);
        d_activation = MathVector(Column, zeros, d_neuronsAmount);

        for (const auto& v: i_initialInput)
            if (v.n_elem > 0)
                d_weights.push_back(
                    RandnMatrix(d_neuronsAmount, v.n_elem));
            else {
                ERROR_LOG("BaseUnit::init : one element in initial list is empty");
                return false;
            }
        return true;
    }

    Column  BasicUnit::forward(const MultipleVector& i_input) {
        Column  result = MathVector(Column, zeros, d_neuronsAmount);
        try {
            result = calcActivation(i_input) + d_bias;
        } catch (std::exception& e) {
            ERROR_LOG("BaseUnit::forward -> " << e.what());
            throw;
        }

        return d_activFunc ? result.transform(d_activFunc) : result;
    }

    Column  BasicUnit::calcActivation(const MultipleVector& i_input) {
        try {
            auto weightIter = d_weights.begin();
            auto inputIter = i_input.begin();
            for (; weightIter != d_weights.end() && inputIter != i_input.end();
                    ++weightIter, ++inputIter)
            {
                d_activation += (*weightIter)*(*inputIter);
            }    
            if (weightIter != d_weights.end() || inputIter != i_input.end())
                throw std::runtime_error ("sizes of inputs and weights list are different");
        } catch (std::exception& e) {
            std::string except("calcAcltivation -> ");
            except += e.what();
            throw std::runtime_error(except);
        }
        return d_activation;
    }

    bool BasicUnit::learn(BasicLearnAlgorithm* i_algorithm) {
        try {
            i_algorithm->start(d_weights, d_bias, d_activation, this);
        } catch (std::exception& e) {
            ERROR_LOG("BasicUnit::learn -> " << e.what());
            return false;
        }
        return true;
    }

}