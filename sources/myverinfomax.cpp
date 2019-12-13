#include    "nnlib.h"


class MyReservoir {

    public:

        MyReservoir(int i_inSize, int i_recSize) {
            d_weights = arma::randu<Matrix>(i_recSize, i_inSize);
            d_recurMatrix = arma::sprandu(i_recSize, i_recSize, 2.0/i_recSize);
            d_state = MathVector(Column, zeros, i_recSize);
            d_bias = MathVector(Column, zeros, i_recSize);
            d_addWeight = MathVector(Column, ones, i_recSize);
        }
        ~MyReservoir(){}

        Column  forward (const Column& i_inp) {
            totalIncomSignal(i_inp);
            out();
            return d_state;
        }

        Column  totalIncomSignal(const Column& i_inp) {
            d_totalIncomSignal = d_weights*i_inp + d_recurMatrix*d_state;
            return d_totalIncomSignal;
        }

        Column  out() {
            d_state = (d_addWeight%d_totalIncomSignal + d_bias);

            return d_state.transform([](double x) {
                return 1.0/(1+exp(-x));
            });
        }

        void learn (const SingleData& i_inpData) {
		    Column mean = calcMean(i_inpData);
	            d_bias = -mean;

		    for (int i = 0; i < d_state.n_elem; ++i) {
		        double m = mean[i];
		        auto theta = [&m](double w)->double{
		            return 1.0 - 1.0/(1 + exp(-w*m + m));
		        };
		        auto f = [&theta, &m](double w)->double{
		            return 1.0/w - (1 - 2*theta(w))*m;
		        };


		        d_addWeight[i] = dihotomia(f);
		//        INFO_LOG("w[" << i << "] = " << d_addWeight[i]);
		//        INFO_LOG("theta : " << i << " = " << theta(d_addWeight[i]));
		    }
        }

	Column calcMean (const SingleData& i_inpData) {
	    Column mean = MathVector(Column, zeros, d_state.n_elem);	
            for (const auto& v : i_inpData) {
                mean += totalIncomSignal(v);
		out();
            }
            return mean / i_inpData.size();
	}

        double dihotomia(std::function<double(double)> f) {
            double a = 0.9;
            double b = 30.0;
            double c = (b+a)/2;
            double fa = f(a);
            double fb = f(b);

            while (fa*fb > 0) {
                b +=10;
                fb = f(b);
            }

            do {
                
                fa = f(a);
                fb = f(b);
                double fc = f(c);

                if (fc == 0) return c;
                if (fc*fa > 0) a = c;
                else b = c;

                c = (a+b)/2;
            }while (b-a > 1e-6 && fa*fb < 0);

            return c;
        }

    private:

        Matrix d_weights;
        SpMatrix    d_recurMatrix;

        Column  d_addWeight;
        Column  d_bias;
        Column  d_state;
        Column  d_totalIncomSignal;
};


class Net {
    public:
        Net(int i_in, int i_r, int i_ou) : d_res(i_in, i_r), d_out(i_ou) {
            d_out.init({MathVector(Column, zeros, i_r)});
        }
        ~Net() {}

        void skip(const DataSet& i_data){
            for(const auto& v: i_data.first)
                d_res.forward(v);
        }
        void learn(const DataSet& i_data){
            d_res.learn(i_data.first);
            MultipleData toOut;
            for(const auto& v: i_data.first)
                toOut.push_back({d_res.forward(v)});
            
            MultipleDataSet mds = {toOut, i_data.second};
            nn::RidgeRegressionAlgorithm alg(mds);
            d_out.learn(&alg);
        }
        void start(const DataSet& i_data){
            SingleData result;
            for (const auto& v: i_data.first) {
                Column h = d_res.forward(v);
                result.push_back(d_out.forward({h}));
            }

            INFO_LOG("nmsre = " << nn::fn::nrmse(result,i_data.second));
        }

    private:

        MyReservoir d_res;
        nn::BasicUnit d_out;
};


// int main (int argc, char* argv[]) {

//     DEBUG_LOG("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n\tTHERE IS SOME BUG\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    
//     try{
       

//         nn::DataSetLoader   loader(argv[1]);
//         DataSet skipDs = loader.form(1500);
//         DataSet learnDs = loader.form(2000);
//         DataSet testDs = loader.form(400);

//         Net net(1, 400, 1);

//         net.skip(skipDs);
//         net.learn(learnDs);
//         net.start(testDs);

       
        
//     } catch (std:: exception& e) {
//         ERROR_LOG(e.what());
//     }
//     return 0;
// }
