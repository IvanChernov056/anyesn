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

    }
}