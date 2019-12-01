#include    "FirstEsnWithBackConnetions.h"



namespace nn {
    
    void    FirstEsnWithBackConnections::init(const MultipleVector& i_initialVector) {
        SingleVector initVec = initUnit(i_initialVector, d_reservoir);
        initUnit({initVec}, d_readout);

        AlgorithmPtr initialCrrection = new BasicLearnAlgorithm();
        learnUnit(initialCrrection, d_reservoir);
        learnUnit(initialCrrection, d_readout);
        delete initialCrrection;
    }

    void    FirstEsnWithBackConnections::start(const MultipleData& i_skipData, const MultipleDataSet& i_learnSet, const DataSet& i_testSet) {
        INFO_LOG("init");
        this->init(i_skipData[0]);
        
        INFO_LOG("skip");
        this->skip(i_skipData);
        
        INFO_LOG("learn");
        this->learn(i_learnSet);
        
        INFO_LOG("test");
        this->test(i_testSet);

        INFO_LOG("end");
    }

    SingleVector    FirstEsnWithBackConnections::initUnit(const MultipleVector& i_initialUnit, BasicUnit* o_unit) {
        try {
            o_unit->init(i_initialUnit);
            return  o_unit->forward(i_initialUnit);
        } catch (std::exception& e) {
            ERROR_LOG ("FirstEsnWithBackConnections::initUnit -> " << e.what());
        }
    }

    void    FirstEsnWithBackConnections::learnUnit(BasicLearnAlgorithm* i_algorithm, BasicUnit* o_unit) {
        try {
            o_unit->learn(i_algorithm);
        } catch (std::exception& e) {
            ERROR_LOG ("FirstEsnWithBackConnections::learnUnit -> " << e.what());
        }
    }

    void    FirstEsnWithBackConnections::createUnits() {
        d_readout = new BasicUnit(2);
        d_reservoir = new BasicReservoir(400);
    }

    void    FirstEsnWithBackConnections::deleteUnits() {
        if(d_reservoir) delete d_reservoir;
        if(d_readout) delete d_readout;
    }


    void    FirstEsnWithBackConnections::skip(const MultipleData& i_data) {
        for(const auto& mpV : i_data)
            d_reservoir->forward(mpV);
    }

    void    FirstEsnWithBackConnections::learn(const MultipleDataSet& i_dataSet) {
        MultipleData resOut;
        for(const auto& mpV : i_dataSet.first)
            resOut.push_back({d_reservoir->forward(mpV)});
        MultipleDataSet mulData={resOut, i_dataSet.second};
        AlgorithmPtr ridgeAlg = new RidgeRegressionAlgorithm(mulData);
        learnUnit(ridgeAlg, d_readout);
        delete ridgeAlg;
        d_etalonForInitPrediction = *(i_dataSet.second.end()-1);
    }

    void    FirstEsnWithBackConnections::test(const DataSet& i_dataSet) {
        Column  backOut = d_etalonForInitPrediction;
        SingleData  result;
        for(const auto& mpV : i_dataSet.first) {
            auto resOut = d_reservoir->forward({mpV, backOut});
            backOut = d_readout->forward({resOut});
            result.push_back(backOut);
        }
    }

}