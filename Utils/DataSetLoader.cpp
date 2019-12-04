#include    "DataSetLoader.h"


namespace nn {

    DataSetLoader::DataSetLoader (const std::string& i_fileName) 
        : d_fileName(i_fileName)
    {
        std::ifstream   file(i_fileName);
        try {
            if (!file.is_open()) {
                throw  std::runtime_error("file does not exist");
            }
            d_fileExist = true;
            file >> d_inputSize >> d_outputSize;
            if (file.bad() || file.fail())
                throw std::runtime_error("fail reading");
            
            d_lineLen = (d_inputSize + d_outputSize)*(15 + 4);
        } catch (std::exception& e) {
            ERROR_LOG("DataSetLoader -> " << e.what());
            file.clear();
            file.close();
            d_fileExist = false;
            d_inputSize = d_outputSize = -1;
            d_lineLen = 1;
        }
        DEBUG_LOG("sizes: " << d_inputSize << '\t' << d_outputSize);
        file.close();
    }

    DataSet DataSetLoader::form(int i_setLenght) {
        return form(i_setLenght, d_startPoint);
    }

    DataSet DataSetLoader::form(int i_setLenght, int i_startPoint) {
        SingleData  etalonSet,  inputSet;

        try {
            if (!d_fileExist) throw std::runtime_error("file does not exist");
            std::ifstream file(d_fileName);
            skipLines(i_startPoint +1, file);
            if (file.eof()) throw std::runtime_error("not enough data");
            d_startPoint = i_startPoint;
            Column  inpVec(d_inputSize), outVec(d_outputSize);
            for (int line = 0; line < i_setLenght; ++line) {
                readLine(file, inpVec, outVec);
                inputSet.push_back(inpVec);
                etalonSet.push_back(outVec);
                if(file.eof()) break;
            }
        } catch(std::exception&e) {
            THROW_FORWARD("DataSetLoader::form -> ", e);
        }

        return {inputSet, etalonSet};
    }

    void    DataSetLoader::readLine(std::istream& io_s, Column& o_inpVec, Column& o_outVec) {
        std::stringstream stream;
        
        char line[d_lineLen];
        
        io_s.getline(line, d_lineLen);
        stream << line;
        readVec(stream, o_inpVec);
        readVec(stream, o_outVec);
        ++d_startPoint;
    }

    void    DataSetLoader::skipLines(int i_linesAmount, std::istream& io_s) {
        if (i_linesAmount <= 0) return;

        for (int line = 0; line < i_linesAmount; ++line) {
            skipLine(io_s);
            if (io_s.eof()) break;
        }
    }

    void    DataSetLoader::readVec(std::istream& io_s, Column& o_vec) {
        for (int i = 0; i < o_vec.n_elem; ++i) {
            io_s >> o_vec[i];
            if (io_s.fail() || io_s.bad())
                throw std::runtime_error("data corupted");
            if (io_s.eof() && i < o_vec.n_elem-1)
                throw std::runtime_error("not enough data in line");
        }
    }
    void    DataSetLoader::skipLine(std::istream& io_s) {
        char line[d_lineLen];
        io_s.getline(line, d_lineLen);
    }
}