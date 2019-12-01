#include    "BasicNetwork.h"

namespace nn {
    BasicNetwork::BasicNetwork () 
        : d_initCorrect(new BasicLearnAlgorithm())
    {
    }

    BasicNetwork::~BasicNetwork() 
    {
    }


    void    BasicNetwork::start(const MultipleData& i_data, const MultipleDataSet& i_learnSet, const DataSet& i_testSet) {
        INFO_LOG("create");
        this->createUnits();
        
        INFO_LOG("init");
        this->init(i_learnSet.first[0]);

        INFO_LOG("learn");
        this->learn(i_learnSet);

        INFO_LOG("test");
        this->test(i_testSet);

        INFO_LOG("end");
    }

    SingleVector    BasicNetwork::initUnit(const MultipleVector& i_initialUnit, UnitPtr o_unit) {
        try {
            o_unit->init(i_initialUnit);
            return  o_unit->forward(i_initialUnit);
        } catch (std::exception& e) {
            ERROR_LOG ("BasicNetwork::initUnit -> " << e.what());
        }
    }

    void    BasicNetwork::learnUnit(BasicLearnAlgorithm* i_algorithm, UnitPtr o_unit) {
        try {
            o_unit->learn(i_algorithm);
        } catch (std::exception& e) {
            ERROR_LOG ("BasicNetwork::learnUnit -> " << e.what());
        }
    }

    
    void    BasicNetwork::init(const MultipleVector& i_initialVecor) 
    {
        //init all units
    }

    void    BasicNetwork::skip(const MultipleData& i_data) 
    {
        //skip part to staiblize recurent units
    }

    void    BasicNetwork::learn(const MultipleDataSet& i_dataSet) {
        //learn all units
    }

    void    BasicNetwork::test(const DataSet& i_dataSet) {
        //test network work
    }

    void    BasicNetwork::createUnits() {
        //create all units
    }

}