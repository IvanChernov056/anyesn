#include    "nnutils.h"



namespace nn {
    namespace fn {
        Matrix    makeMatrixFromContainer(const ColumnContainer& i_container) {
            Matrix  result(i_container[0].size(), 1);
            try {
                THROW_EMPTY(i_container);
                CONCATINATE_LIST(horiz, i_container, result);
            } catch (std::exception& e) {
                THROW_FORWARD("makeMatrixFromContainer -> ", e);
            }
            
            return result;
        }

        Column    uniteMultipeVector(const MultipleVector& i_inp) {
            Column result;
            try {
                THROW_EMPTY(i_inp);
                CONCATINATE_LIST(vert, i_inp, result);
            } catch (std::exception& e) {
                THROW_FORWARD("uniteMultipeVector -> ", e);
            }
            return result;
        }

        void      splitMatrixToMuliple (MultipleWeight& o_multiWeight, const Matrix& i_toSplit, const std::vector<int>& i_range) {
            try {
                int leftPoint = 0;
                int rightPoint = 0;
                for (int i = 0; i < i_range.size(); ++i) {
                    rightPoint += i_range[i];
                    o_multiWeight[i] = i_toSplit.cols(leftPoint, rightPoint-1);
                    leftPoint += i_range[i];
                }    
            } catch (std::exception& e) {
                THROW_FORWARD("splitMatrixToMuliple -> ", e);
            }
        }

        void      splitMatrixToMuliple (MultipleWeight& o_multiWeight, const Matrix& i_toSplit, const MultipleVector& i_rangeVector) {
            std::vector<int> range;
            for (auto& v : i_rangeVector) 
                range.push_back(v.size());
            splitMatrixToMuliple(o_multiWeight, i_toSplit, range);
        }


        Matrix    makeMatrixFromMultipleData(const MultipleData& i_multiData) {
            Matrix result;
            try {
                ColumnContainer longVectorList;
                for (const auto& mulVec : i_multiData)
                    longVectorList.push_back(uniteMultipeVector(mulVec));
                result = makeMatrixFromContainer(longVectorList);
            } catch (std::exception& e) {
                THROW_FORWARD("makeMatrixFromMultipleData -> ", e);
            }    
            return result;
        }

        Matrix    makeCovarianceMatrix (const Matrix& i_inpMat) {
            Matrix result = i_inpMat;
            
            Column  mean = MathVector(Column, zeros, i_inpMat.n_rows);
            i_inpMat.each_col([&mean](const Column& v){mean += v;});
            mean /= i_inpMat.n_cols;

            result.each_col([&mean](Column& v){v -= mean;});

            result = result*result.t();
            return i_inpMat.n_cols > 1 ? result/(i_inpMat.n_cols-1) : result;
        }

        void      bindSingleToMultiple(const SingleData& i_singlesList, MultipleData& io_mulData) {
            auto mulIetr = io_mulData.begin();
            auto singIter = i_singlesList.begin();
            for(; mulIetr!=io_mulData.end() && singIter!=i_singlesList.end(); 
                    ++mulIetr, ++singIter)
            {
                mulIetr->push_back(*singIter);
            }
        }
   
    MultipleData zip(const SingleData& i_inp1, const SingleData& i_inp2) {
        MultipleData result;
        auto it1 = i_inp1.begin();
        auto it2 = i_inp2.begin();
        for (;it1 != i_inp1.end() && it2!=i_inp2.end(); ++it1, ++it2)
            result.push_back({*it1, *it2});
        return result;
    }

    double    squaredEuclideanNorom (const Column& i_inp) {
        double  nrm = 0;
        i_inp.for_each([&nrm](double x){nrm += x*x;});
        return nrm;
    }

    double    nrmse (const SingleData& i_predicted, const SingleData& i_etalon) {
        if (i_predicted.size() < i_etalon.size())
            throw std::runtime_error("nrmse -> not enough etalon data");
        
        SingleVector avrgEtalon = average(i_etalon);
        DoubleContainer numeratorList, denumeratorList;

        auto predIter = i_predicted.begin();
        auto etalIter = i_etalon.begin();
        for (;predIter!=i_predicted.end() && etalIter!=i_etalon.end(); ++predIter, ++etalIter) {
            numeratorList.push_back(squaredEuclideanNorom(*predIter - *etalIter));
            denumeratorList.push_back(squaredEuclideanNorom(*predIter - avrgEtalon));
        }

        double numerator = average(numeratorList);
        double denumerator = average(denumeratorList);

        if (denumerator <= 0)
            throw std::runtime_error("nrmse -> prediction failed or input data did not have any difference");
        
        return sqrt(numerator / denumerator);
    }



    }
}