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


    }
}