#ifndef     ABSTRACT_RECURSIVE_UNIT_H
#define     ABSTRACT_RECURSIVE_UNIT_H

#include    "AbstractForwardUnit.h"

namespace nn {
    class AbstractRecursiveUnit : public AbstractForwardUnit {

        public:

            virtual Column operator()(const Column& i_input, const Column& i_returned);
            virtual Column operator()(const Column& i_input) override;

        protected:

            Column  d_state;
    };
}

#endif