#ifndef     BASIC_RESERVOIR_H
#define     BASIC_RESERVOIR_H


#include    "BasicUnit.h"

namespace nn {

    class BasicReservoir : public BasicUnit {

        public:

            BasicReservoir (int i_neuronsAmount, double i_density = 0.03, double i_radius = 1.0, Activation i_func = nullptr);
            virtual ~BasicReservoir(){}

            virtual Column  forward(const MultipleVector& i_initialInput) override;
            virtual bool learn(BasicLearnAlgorithm* i_algorithm) override;

        protected:

            virtual const Column&  totalIncomingSignal(const MultipleVector& i_inputSignales) override;

            SpMatrix    d_recurMatrix;
            Column      d_state;
            double		d_leakRate;
    };
}


#endif