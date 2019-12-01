#ifndef     FORWARD_ESN_H
#define     FORWARD_ESN_H

#include    "BasicNetwork.h"

namespace nn {
    class ForwardEsn : public BasicNetwork {
        public:
            ForwardEsn(UnitsList i_units);
            ~ForwardEsn();

        protected:

            virtual void    init(const MultipleVector& i_initialVecor) override;
            virtual void    skip(const MultipleData& i_data) override;
            virtual void    learn(const MultipleDataSet& i_dataSet) override;
            virtual void    test(const DataSet& i_dataSet) override;
            virtual void    createUnits() override;

        protected:

            BasicUnit*     d_reservoir;
            BasicUnit*     d_readout;

    };
}

#endif
