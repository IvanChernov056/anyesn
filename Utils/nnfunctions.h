#ifndef     NN_FUNCTIONS_H
#define     NN_FUNCTIONS_H


#include    "nntypes.h"


namespace nn {
    namespace fn {
        Matrix    makeMatrixFromContainer(const ColumnContainer& i_container);
        Matrix    makeMatrixFromMultipleData(const MultipleData& i_multiData);
        Matrix    makeCovarianceMatrix (const Matrix& i_inpMat);
        Column    uniteMultipeVector(const MultipleVector& i_input);
        void      bindSingleToMultiple(SingleData& i_singlesList, MultipleData& io_mulData);
        void      splitMatrixToMuliple (MultipleWeight& o_multiWeight, const Matrix& i_toSplit, const std::vector<int>& i_range);
        void      splitMatrixToMuliple (MultipleWeight& o_multiWeight, const Matrix& i_toSplit, const MultipleVector& i_rangeVector);
        MultipleData zip(const SingleData& i_inp1, const SingleData& i_inp2);
        double    squaredEuclideanNorom (const Column& i_inp);
        double    nrmse (const SingleData& i_predicted, const SingleData& i_etalon);


        template<class T>
        T   average (const std::vector<T>& i_list) {
            if(i_list.empty()) 
                return T();

            T   avrg = i_list[0];
            for (auto it = i_list.begin()+1; it != i_list.end(); ++it)
                avrg += *it;
            return avrg / i_list.size();
        }
    }
}

#endif