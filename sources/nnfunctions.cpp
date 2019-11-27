#include    "nnutils.h"


// namespace nn {
//     namespace fn {
//         Data    formDataFromContainer(const ColumnContainer& i_container) {
//             if (i_container.empty()) return Data();

//             Data data(i_container[0]);
//             for (auto it = i_container.begin()+1; it != i_container.end(); ++it)
//                 data = arma::join_horiz (data, *it);
            
//             return data;
//         }
//     }
// }