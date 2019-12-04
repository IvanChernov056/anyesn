#include    "BasicLearnAlgorithm.h"
#include    "BasicUnit.h"


namespace nn {
   
    void BasicLearnAlgorithm::start(MultipleWeight& io_weights, Column& io_bias, Column& i_totalIncomingSignal, BasicUnit *i_unit){
        Column  mean = MathVector(Column, zeros, io_bias.n_elem);
        int colNumber = 0;
        for (auto& w : io_weights) {
            double wNorm = NORM2(w);
            if (wNorm > 0) {
                w /= wNorm*io_weights.size();
                colNumber += w.n_cols;
                w.each_col([&mean](const Column& v)
                        {mean += v;});
            }
            else throw std::runtime_error(
                        "BaseReservoirCorrectAlgorithm::start -> trying to correct not initialized weights list");
        }

        io_bias = -mean / colNumber;
    }

}