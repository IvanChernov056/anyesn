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


#define     THROW_EMPTY(LIST) if (LIST.empty()) throw std::runtime_error("input list is empty");
#define     CONCATINATE(TYPE, LEFT_UP, RIGHT_DOWN)\
                arma::join_##TYPE(LEFT_UP, RIGHT_DOWN)

#define     CONCATINATE_LIST(TYPE, LIST, RESULT) \
                RESULT = LIST[0];\
                for (auto it = LIST.begin()+1; it != LIST.end(); ++it)\
                    RESULT = CONCATINATE(TYPE, RESULT, *it);

#define     THROW_FORWARD(msg, EXCEPTION) \
                std::string exceptionMsg(msg);\
                exceptionMsg += EXCEPTION.what();\
                throw std::runtime_error(exceptionMsg);


#define     EYE(ELEM_COUNT) Matrix(ELEM_COUNT, ELEM_COUNT, arma::fill::eye)
#define     INV(SQ_MAT) arma::inv(SQ_MAT)
#define     PINV(SQ_MAT) arma::pinv(SQ_MAT)
#define     INV_SYMPD(SQ_MAT) arma::inv_sympd(SQ_MAT)

#endif