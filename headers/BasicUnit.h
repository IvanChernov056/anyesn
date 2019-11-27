#ifndef     BASIC_UNIT_H
#define     BASIC_UNIT_H


#include    "nnutils.h"

namespace nn {

    class BasicUnit {
        public:

            explicit BasicUnit(int i_neuronsAmount, Activation i_func = nullptr);
            virtual ~BasicUnit();

            virtual bool init(const MultipleVector& i_initialInput);
            virtual Column  forward(const MultipleVector& i_initialInput);

            //virtual bool learn(/*some param*/){return true;}
            ////now i don't know how it sould be done.

        protected:

            MultipleWeight  d_weights;
            Activation      d_activFunc;
            int             d_neuronsAmount;
    };
}

#endif