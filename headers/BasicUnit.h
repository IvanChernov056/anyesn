#ifndef     BASIC_UNIT_H
#define     BASIC_UNIT_H


#include    "nnutils.h"

namespace nn {

    class   BasicLearnAlgorithm;

    class BasicUnit {
        public:

            explicit BasicUnit(int i_neuronsAmount, Activation i_func = nullptr);
            virtual ~BasicUnit();

            virtual bool init(const MultipleVector& i_initialInput);
            virtual Column  forward(const MultipleVector& i_initialInput);

            virtual bool learn(BasicLearnAlgorithm* i_algorithm);
            

        protected:

            virtual Column  calcActivation(const MultipleVector& i_initialInput);



            Column          d_bias;
            Column          d_activation;
            MultipleWeight  d_weights;
            Activation      d_activFunc;
            int             d_neuronsAmount;
    };
}

#endif