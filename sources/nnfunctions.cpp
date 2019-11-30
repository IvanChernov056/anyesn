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
   
    }
}