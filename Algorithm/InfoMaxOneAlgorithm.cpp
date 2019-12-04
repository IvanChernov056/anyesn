#include    "InfoMaxOneAlgorithm.h"


namespace nn {

    InfoMaxOneAlgorithm::InfoMaxOneAlgorithm(const MultipleData& i_data, int i_iterations, double i_learnSpead) 
        : d_data(i_data), d_iterations(i_iterations), d_learnSpead(i_learnSpead)
    {
    }

    InfoMaxOneAlgorithm::~InfoMaxOneAlgorithm() {

    }

    void InfoMaxOneAlgorithm::start(MultipleWeight& io_weights, Column& io_bias, Column& io_totalIncomingSignal, BasicUnit *i_unit) {
        Column  out;
        d_tisGain = MathVector(Column, randu, io_bias.n_elem);
        d_optBias = MathVector(Column, randn, io_bias.n_elem);


        for (int iter = 0; iter < d_iterations; ++iter) {
            for (const auto& mpV: d_data) {
               out = this->unitOut(i_unit, mpV, io_bias, io_totalIncomingSignal);
               d_optBias -= deltaOptBias(io_totalIncomingSignal, out) * d_learnSpead;
               d_tisGain -= deltaTisGain() * d_learnSpead;
            }

            d_learnSpead *= 0.9;
            INFO_LOG("\t iter = " << iter);
        }
    }

    Column InfoMaxOneAlgorithm::unitOut(BasicUnit* i_unit, const MultipleVector& i_inp, Column& io_bias, Column& io_totalIncomingSignal) {
        i_unit->totalIncomingSignal(i_inp);
        
        io_totalIncomingSignal = io_totalIncomingSignal%d_tisGain;
        
        DEBUG_LOG("opt: " << NORM2(io_bias));
        io_bias -= d_optBias;
        DEBUG_LOG("opt: " << NORM2(io_bias) << '\n');
        return i_unit->out();
    }

    const Column& InfoMaxOneAlgorithm::deltaTisGain() {
        d_deltaTisGain = d_tisGain;
        d_deltaTisGain.for_each([](double& w) {
            if (w!=0)
                w = 1.0/w;
        });

        d_deltaTisGain += d_deltaOptBias;
        return d_deltaTisGain;
    }

    const Column& InfoMaxOneAlgorithm::deltaOptBias(const Column& i_unitInp, const Column& i_unitOut) {
        d_deltaOptBias = i_unitInp%(MathVector(Column, ones, i_unitInp.n_elem) - 2*i_unitOut);
        return d_deltaOptBias;
    }

}