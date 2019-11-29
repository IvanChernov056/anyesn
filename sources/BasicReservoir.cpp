#include    "BasicReservoir.h"


namespace nn {

    BasicReservoir::BasicReservoir (int i_neuronsAmount, double i_density, double i_radius, Activation i_func) 
        : BasicUnit(i_neuronsAmount, i_func), d_state(MathVector(Column, zeros, i_neuronsAmount))
    {
        d_recurMatrix = RandnSpMatrix(d_neuronsAmount, d_neuronsAmount, i_density);
        double  norm = NORM2(d_recurMatrix);
        if (norm > 0)
            d_recurMatrix *= i_radius/norm;
    }

    Column  BasicReservoir::forward(const MultipleVector& i_inputSignales) {
        try {
            d_state = this->totalIncomingSignal(i_inputSignales) + d_bias;
            if (d_activFunc) d_state.transform(d_activFunc);
        } catch (std::exception& e) {
            THROW_FORWARD("BasicReservoir::forward -> ", e);
        }
        return d_state;
    }


    bool BasicReservoir::learn(BasicLearnAlgorithm* i_algorithm) {
        try {
            BasicUnit::learn(i_algorithm);
        } catch(std::exception& e) {
            THROW_FORWARD("BasicReservoir::learn -> ", e);
        }
        return true;
    }


    const Column&  BasicReservoir::totalIncomingSignal(const MultipleVector& i_inputSignales) {
        d_totalIncomingSignal = BasicUnit::totalIncomingSignal(i_inputSignales) + d_recurMatrix*d_state;
        return d_totalIncomingSignal;
    }

}