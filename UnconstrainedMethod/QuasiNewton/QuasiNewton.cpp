#include <Msnhnet/math/MsnhMatrixS.h>
#include <Msnhnet/cv/MsnhCVGui.h>
#include <iostream>

using namespace Msnhnet;

enum Method
{
    DFP,
    BFGS
};

class QuasiNewton
{
public:
    QuasiNewton(int maxIter, double eps, double rho, double tau, Method method):_maxIter(maxIter),_eps(eps),_rho(rho),_tau(tau),_method(method){}

    void setMaxIter(int maxIter)
    {
        _maxIter = maxIter;
    }

    virtual int solve(MatSDS &startPoint) = 0;

    void setEps(double eps)
    {
        _eps = eps;
    }

    void setRho(double rho)
    {
        _rho = rho;
    }

    void setTau(double tau)
    {
        _tau = tau;
    }

    void setMethod(const Method &method)
    {
        _method = method;
    }


    //正定性判定
    bool isPosMat(const MatSDS &H)
    {
        MatSDS eigen = H.eigen()[0];
        for (int i = 0; i < eigen.mWidth; ++i)
        {
            if(eigen[i]<=0)
            {
                return false;
            }
        }

        return true;
    }

    const std::vector<Vec2F32> &getXStep() const
    {
        return _xStep;
    }

protected:
    int _maxIter = 100;
    double _eps = 0.00001;
    double _rho = 0.2;
    double _tau = 0.9;
    Method _method;
    bool _firstStep = false;
    MatSDS _Dk;
    std::vector<Vec2F32> _xStep;
protected:
    virtual MatSDS calGradient(const MatSDS& point) = 0;
    virtual MatSDS function(const MatSDS& point) = 0;
};


class QuasiNewtonProblem1:public QuasiNewton
{
public:
    QuasiNewtonProblem1(int maxIter, double eps, double rho, double tau, Method method):QuasiNewton(maxIter, eps, rho,tau,method){}

    MatSDS calGradient(const MatSDS &point) override
    {
        MatSDS J(1,2);
        double x1 = point(0,0);
        double x2 = point(0,1);

        J(0,0) = 6*x1 - 2*x1*x2;
        J(0,1) = 6*x2 - x1*x1;

        return J;
    }


    MatSDS function(const MatSDS &point) override
    {
        MatSDS f(1,1);
        double x1 = point(0,0);
        double x2 = point(0,1);

        f(0,0) = 3*x1*x1 + 3*x2*x2 - x1*x1*x2;

        return f;
    }

    int solve(MatSDS &startPoint) override
    {
        MatSDS x = startPoint;

        _firstStep = true;

        for (int i = 0; i < _maxIter; ++i)
        {
            _xStep.push_back({(float)x[0],(float)x[1]});
            MatSDS J = calGradient(x);

            if(_firstStep)
            {
                _Dk = MatSDS::eye(x.mHeight);
                _firstStep = false;
            }

            MatSDS dk = -1*_Dk.invert()*J;

            std::cout<<std::left<<"Iter(s): "<<std::setw(4)<<i<<", Loss: "<<std::setw(12)<<dk.L2()<<" Result: "<<function(x)[0]<<std::endl;


            if(dk.LInf() < _eps)
            {
                startPoint = x;
                return i+1;
            }

            double alpha = 1;

            //Armijo准则
            for (int i = 0; i < 100; ++i)
            {
                MatSDS left  = function(x + alpha*dk);
                MatSDS right = function(x) + this->_rho*alpha*calGradient(x).transpose()*dk;

                if(left(0,0) <= right(0,0))
                {
                    break;
                }

                alpha = alpha * _tau;
            }

            MatSDS sk = alpha*dk;

            x = x + sk;

            MatSDS J1 = calGradient(x);

            MatSDS yk = J1 - J;


            if(_method == BFGS)
            {
                MatSDS skT = sk.transpose();
                MatSDS ykT = yk.transpose();

                _Dk = _Dk + (yk*ykT)/((ykT*sk)(0,0)) - (_Dk*sk*skT*_Dk)/((skT*_Dk*sk)(0,0));
            }
            else if(_method == DFP)
            {
                MatSDS skT = sk.transpose();
                MatSDS ykT = yk.transpose();

                MatSDS DkTmp =_Dk.invert();

                _Dk = DkTmp + (sk*skT)/((skT*yk)(0,0)) - (DkTmp*yk*ykT*DkTmp)/((ykT*DkTmp*yk)(0,0));

                _Dk = _Dk.invert();
            }

        }

        return -1;
    }
};



int main()
{
    QuasiNewtonProblem1 function(1000, 0.001, 0.4, 0.8, BFGS);
    MatSDS startPoint(1,2,{4,3}); //牛顿法，DFP无解,BFGS有解

    try
    {
        int res = function.solve(startPoint);
        if(res < 0)
        {
            std::cout<<"求解失败"<<std::endl;
        }
        else
        {
            std::cout<<"求解成功! 迭代次数: "<<res<<std::endl;
            std::cout<<"最小值点："<<res<<std::endl;
            startPoint.print();

            std::cout<<"此时方程的值为："<<std::endl;
            function.function(startPoint).print();

#ifdef WIN32
        Gui::setFont("c:/windows/fonts/MSYH.TTC",16);
#endif
        std::cout<<"按\"esc\"退出!"<<std::endl;
        Gui::plotLine(u8"拟牛顿法迭代X中间值","x",function.getXStep());
        Gui::wait();
        }

    }
    catch(Exception ex)
    {
        std::cout<<ex.what();
    }
}
