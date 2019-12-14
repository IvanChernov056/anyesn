#include    "BasicNetwork.h"


namespace nn {
    class SecondForwardEsn : public BasicNetwork {
        public:
            SecondForwardEsn(UnitsList i_units);
            ~SecondForwardEsn();

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

namespace nn {
    SecondForwardEsn::SecondForwardEsn(UnitsList i_units) try {
        if (i_units.size()!=2)
            throw std::runtime_error("in this constructor must be 2 units");
        d_reservoir = i_units[0];
        d_readout = i_units[1];
    } catch (std::exception& e) {
        THROW_FORWARD("SecondForwardEsn -> ", e);
    }

    SecondForwardEsn::~SecondForwardEsn(){
        delete d_readout;
        delete d_reservoir;
    }

    void    SecondForwardEsn::init(const MultipleVector& i_initialVecor) {
        d_reservoir->init(i_initialVecor);
        Column resOut = d_reservoir->forward(i_initialVecor);
        d_readout->init({resOut});
    }
    
    void    SecondForwardEsn::skip(const MultipleData& i_data) {
        for(const auto& mpV : i_data)
            d_reservoir->forward(mpV);
    }
    
    void    SecondForwardEsn::learn(const MultipleDataSet& i_dataSet) {
        AlgorithmPtr infomax(new InfoMaxOneAlgorithm(i_dataSet.first, 10, 0.04));
        // d_reservoir->learn(infomax.get());

        MultipleData resOut;
        for(const auto& mpV : i_dataSet.first)
            resOut.push_back({d_reservoir->forward(mpV)});
        
        MultipleDataSet rdoutSet{resOut, i_dataSet.second};
        AlgorithmPtr ridgeAlg(new RidgeRegressionAlgorithm(rdoutSet));
        d_readout->learn(ridgeAlg.get());
    }
    
    void    SecondForwardEsn::test(const DataSet& i_dataSet) {
        MultipleData resOut;
        for(const auto& mpV : i_dataSet.first)
            resOut.push_back({d_reservoir->forward({mpV})});
        
        SingleData result;
        for(const auto& mpV : resOut)
            result.push_back(d_readout->forward({mpV}));
        
        INFO_LOG("nrmse : " << fn::nrmse(result, i_dataSet.second));
    }
    
    void    SecondForwardEsn::createUnits() {

    }
    
}


// int main (int argc, char* argv[]) {

//     DEBUG_LOG("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n\tTHERE IS SOME BUG\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    
//     try{
//         nn::UnitsList units 
//             {
//                new nn::BasicReservoir(400, 0.03, 1.2, [](double x)->double{return 1.0/(1+exp(-x));}),
//                new nn::BasicUnit(1)
//             };
        
//         nn::SecondForwardEsn net(units);

//         nn::DataSetLoader   loader(argv[1]);
//         DataSet skipDs = loader.form(150);
//         DataSet learnDs = loader.form(2000);
//         DataSet testDs = loader.form(400);

//         MultipleData skipMul;
//         for (const auto& v: skipDs.first)
//             skipMul.push_back({v});

//         MultipleData learnMul;
//         for (const auto& v: learnDs.first)
//             learnMul.push_back({v});
//         MultipleDataSet learnMulDs{learnMul, learnDs.second};

//         net.start(skipMul, learnMulDs, testDs);
        
//     } catch (std:: exception& e) {
//         ERROR_LOG(e.what());
//     }
//     return 0;
// }