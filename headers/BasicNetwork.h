#ifndef     BASIC_NETWORK_H
#define     BASIC_NETWORK_H


#include    "nnlib.h"

namespace nn {
    using   UnitPtr = std::unique_ptr<BasicUnit>;
    using   AlgorithmPtr = std::unique_ptr<BasicLearnAlgorithm>;
    using   UnitsList = std::vector<BasicUnit*>;


    class BasicNetwork {
        public:
            BasicNetwork (/*UnitsList& i_units --> this must be in child */);
            virtual ~BasicNetwork();


            void    start(const MultipleData& i_data, const MultipleDataSet& i_learnSet, const DataSet& i_testSet);
        
        protected:

            SingleVector    initUnit(const MultipleVector& i_initialUnit, UnitPtr o_unit);
            void            learnUnit(BasicLearnAlgorithm* i_algorithm, UnitPtr o_unit);
            void            addUnit(AlgorithmPtr* i_newUnit);
            
            virtual void    init(const MultipleVector& i_initialVecor);
            virtual void    skip(const MultipleData& i_data);
            virtual void    learn(const MultipleDataSet& i_dataSet);
            virtual void    test(const DataSet& i_dataSet);
            virtual void    createUnits();

        protected:

            AlgorithmPtr    d_initCorrect;
    };
}

#endif