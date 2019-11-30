#ifndef     FIRST_ESN_WITH_BACK_CONNECTIONS_H
#define     FIRST_ESN_WITH_BACK_CONNECTIONS_H

#include    "nnlib.h"


namespace nn {
    
    using   UnitPtr = BasicUnit*;
    using   AlgorithmPtr = BasicLearnAlgorithm*;

    class FirstEsnWithBackConnections {

        public:
            FirstEsnWithBackConnections() {createUnits();}
            ~FirstEsnWithBackConnections(){deleteUnits();}
            
            void    init(const MultipleVector& i_initialVecor);
            void    start(const MultipleData& i_data, const DataSet& i_learnSet, const DataSet& i_testSet);

        private:
    
            void    skip(const MultipleData& i_data);
            void    learn(const DataSet& i_dataSet);
            void    test(const DataSet& i_dataSet);
            void    createUnits();
            void    deleteUnits();
            SingleVector    initUnit(const MultipleVector& i_initialUnit, UnitPtr o_unit);
            void    learnUnit(BasicLearnAlgorithm* i_algorithm, UnitPtr o_unit);

        private:

            UnitPtr d_reservoir{nullptr};
            UnitPtr d_readout{nullptr};
            Column  d_etalonForInitPrediction;
    };
}

#endif