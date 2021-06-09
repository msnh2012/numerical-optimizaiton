#include <iostream>
#include <Msnhnet/Msnhnet.h>

using namespace Msnhnet;

class LevenbergMarquardtProblemSmallJac
{
public:
    LevenbergMarquardtProblemSmallJac(int maxIter, double eps, const Mat& x, const Mat& y):_maxIter(maxIter),_eps(eps),_x(x),_y(y){}

    void setMaxIter(int maxIter)
    {
        _maxIter = maxIter;
    }

    void setEps(double eps)
    {
        _eps = eps;
    }

    void setX(const Mat &x)
    {
        _x = x;
    }

    void setY(const Mat &y)
    {
        _y = y;
    }

    Mat calJacobian(const double &a, const double&b, const double&x)
    {
        double eps = 1e-6;

        double gradA = (function(a + eps,b,x) - function(a - eps,b,x))/(2*eps);
        double gradB = (function(a,b + eps,x) - function(a,b- eps, x))/(2*eps);

        Mat J(1,2,MAT_GRAY_F64);

        J.setListData<double>({gradA,gradB});
        return J;
    }

    double function(const double &a, const double&b, const double&x)
    {
        return a*x/(b+x);
    }

    std::vector<double> function(const double &a, const double&b, const std::vector<double>& x)
    {
        std::vector<double> y;
        for (int i = 0; i < x.size(); ++i)
        {
            y.push_back(a*x[i]/(b+x[i]));
        }
        return y;
    }

    int solve(Mat &point)
    {
        Mat pointNew = point;

        Mat eye = Mat::eye(point.getHeight(), MAT_GRAY_F64);

        for (int i = 0; i < _maxIter; ++i)
        {

            point = pointNew;

            double a = point.getPixel<double>({0,0});
            double b = point.getPixel<double>({0,1});

            Mat H(2,2,MAT_GRAY_F64);
            Mat Jfx(1,2,MAT_GRAY_F64);
            Mat J(1,2,MAT_GRAY_F64);


            for (int j = 0; j < _x.getHeight(); ++j)
            {
                double x  = _x.getFloat64()[j];

                Mat J_ = calJacobian(a,b,x);

                J = J + J_;

                double fx = function(a,b,x) -  _y.getFloat64()[j];

                H   = H  + J_*J_.transpose();
                Jfx = Jfx + J_*fx;
            }

            if(i == 0)
            {
                std::vector<double> diag = H.getDiagList<double>();
                _u = ExVector::max<double>(diag);
            }

            Mat dk = -(H + _u*eye).pseudoInvert()*Jfx;

            double da = dk.getPixel<double>({0,0});
            double db = dk.getPixel<double>({0,1});

            double errNorm = 0;

            for (int j = 0; j < _x.getHeight(); ++j)
            {
                double x  = _x.getFloat64()[j];
                double dt = function(a+da,b+db,x) - function(a,b,x);
                errNorm = errNorm + sqrt(dt*dt);
            }

            double rho = errNorm/((J.transpose()*dk).norm(NormType::NORM_L2));//????

            if(rho > 0)
            {
                double t = 2*rho - 1;
                _u = _u * (std::max)(1/3.0, 1-t*t*t);
                _v = 2;
                pointNew = point + dk;
            }
            else
            {
                _u = _u*_v;
                _v = 2*_v;
            }

            if((pointNew-point).norm(NormType::NORM_L2) < _eps)
            {
                point = pointNew;
                return i;
            }
        }
    }

private:
    int _maxIter = 100;
    double _eps = 0.00001;
    double _v   = 2;
    double _u   = 0;
    Mat _x;
    Mat _y;
};

// y = a*x/(b+x)
int main()
{
    std::vector<double> x;
    std::vector<double> y;

    IO::readVector<double>(x,"D:/GN/x.dat","\n");
    IO::readVector<double>(y,"D:/GN/y.dat","\n");

    assert(x.size()==y.size());

    std::vector<Vec2F32> datas;

    for (int i = 0; i < x.size(); ++i)
    {
        datas.push_back(Vec2F32(x[i],y[i]));
    }

#ifdef WIN32
    Gui::setFont("c:/windows/fonts/MSYH.TTC",16);
#endif
    std::cout<<"按\"esc\"退出!"<<std::endl;
    Gui::plotPoints(u8"LM小雅克比法",u8"数据",datas);


    Mat matX(1,x.size(),MAT_GRAY_F64);
    matX.setListData<double>(x);

    Mat matY(1,y.size(),MAT_GRAY_F64);
    matY.setListData<double>(y);

    LevenbergMarquardtProblemSmallJac fun(100,0.00001,matX,matY);

    Mat startPoint(1,2,MAT_GRAY_F64);
    startPoint.setListData<double>({5,1});

    std::cout<<"迭代次数: "<<fun.solve(startPoint)<<std::endl;;
    std::cout<<"系数: "<<std::endl;

    startPoint.print();

    double a = startPoint.getListData<double>()[0];
    double b = startPoint.getListData<double>()[1];

    std::vector<double> xTest = Mat::linspace<double>(0,5,100);
    std::vector<double> yTest = fun.function(a,b,xTest);

    datas.clear();
    for (int i = 0; i < xTest.size(); ++i)
    {
        datas.push_back(Vec2F32(xTest[i],yTest[i]));
    }
    Gui::plotLine(u8"LM小雅克比法",u8"拟合曲线",datas);
    Gui::wait();

}

