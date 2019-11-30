#ifndef     DATA_SET_LOADER_H
#define     DATA_SET_LOADER_H


#include    "nnutils.h"

namespace nn {
    class   DataSetLoader {
        public:
            DataSetLoader (const std::string& i_fileName);
            DataSet form(int i_setLenght);
            DataSet form(int i_setLenght, int i_startPoint);

        private:

            void    readLine(std::istream& io_s, Column& o_inpVec, Column& o_outVec);
            void    skipLines(int i_linesAmount, std::istream& io_s);
            void    skipLine(std::istream& io_s);
            void    readVec(std::istream& io_s, Column& o_vec);

        private:

            int d_startPoint{0};
            int d_inputSize;
            int d_outputSize;
            bool d_fileExist;
            int d_lineLen;
            std::string d_fileName;
    };
}

#endif