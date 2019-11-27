#ifndef     UTILS_H
#define     UTILS_H

#include    "nntypes.h"
#include    "nnfunctions.h"

#include    <iostream>
#include    <fstream>
#include    <sstream>
#include    <string>
#include    <omp.h>

#include    "timer.h"


#define     BASE_LOGGER(logger, msg) logger << msg << '\n'
#define     INFO_LOG(msg)      BASE_LOGGER(std::cout, msg)
#define     ERROR_LOG(msg)     BASE_LOGGER(std::cerr, "ERROR> " << msg)
#define     DEBUG_LOG(msg)     INFO_LOG("DEBUG> " << msg)

#define     RandnMatrix(rows, cols) arma::randn<Matrix>(rows, cols)
#define     RandnVector(TYPE, elem) arma::randn<TYPE>(elem)
#define     RandnSpMatrix(rows, cols, density) arma::sprandn(rows, cols, density)
#define     MathVector(TYPE, FILLER, ELEM_COUNT) TYPE(ELEM_COUNT, arma::fill::FILLER)

#endif