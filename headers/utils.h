#ifndef     UTILS_H
#define     UTILS_H


#include    "types.h"

#define     BASE_LOGGER(logger, msg) logger << msg << '\n'
#define     INFO_LOG(msg)      BASE_LOGGER(std::cout, msg)
#define     ERROR_LOG(msg)     BASE_LOGGER(std::cerr, msg)

#define     RandnMatrix(rows, cols) arma::randn<Matrix>(rows, cols)
#define     RandnVector(TYPE, elem) arma::randn<TYPE>(elem)

#endif