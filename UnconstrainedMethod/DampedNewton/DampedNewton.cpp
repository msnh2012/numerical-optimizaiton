#include <Msnhnet/math/MsnhMatrixS.h>
#include <Msnhnet/cv/MsnhCVGui.h>
#include <iostream>

using namespace Msnhnet;

class DampedNewton
{
public:
    DampedNewton(int maxIter, double eps, double rho, double tau):_maxIter(maxIter),_eps(eps),_rho(rho),_tau(tau){}


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
    std::vector<Vec2F32> _xStep;
protected:
    virtual MatSDS calGradient(const MatSDS& point) = 0;
    virtual MatSDS calHessian(const MatSDS& point) = 0;
    virtual bool calDk(const MatSDS& point, MatSDS &dk) = 0;
    virtual MatSDS function(const MatSDS& point) = 0;
};


class DampedNewtonProblem1:public DampedNewton
{
public:
    DampedNewtonProblem1(int maxIter, double eps, double rho, double tau):DampedNewton(maxIter, eps, rho, tau){}

    MatSDS calGradient(const MatSDS &point) override
    {
        MatSDS J(1,2);
        double x1 = point(0,0);
        double x2 = point(0,1);

        J(0,0) = 6*x1 - 2*x1*x2;
        J(0,1) = 6*x2 - x1*x1;

        return J;
    }

    MatSDS calHessian(const MatSDS &point) override
    {
        MatSDS H(2,2);
        double x1 = point(0,0);
        double x2 = point(0,1);

        H(0,0) = 6 - 2*x2;
        H(0,1) = -2*x1;
        H(1,0) = -2*x1;
        H(1,1) = 6;

        return H;
    }


    bool calDk(const MatSDS& point, MatSDS &dk) override
    {
        MatSDS J = calGradient(point);
        MatSDS H = calHessian(point);
        if(!isPosMat(H))
        {
            return false;
        }
        dk = -1*H.invert()*J;
        return true;
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
        for (int i = 0; i < _maxIter; ++i)
        {

            _xStep.push_back({(float)x[0],(float)x[1]});
            MatSDS dk;

            bool ok = calDk(x, dk);

            if(!ok)
            {
                return -2;
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

            x = x + alpha*dk;

            if(dk.LInf() < _eps)
            {
                startPoint = x;
                return i;
            }
        }

        return -1;
    }
};



int main()
{
    DampedNewtonProblem1 function(100, 0.001, 0.4, 0.8);
    MatSDS startPoint(1,2,{1.5,1.5});

    try
    {
        int res = function.solve(startPoint);
        if(res == -1)
        {
            std::cout<<"求解失败"<<std::endl;
        }
        else if(res == -2)
        {
            std::cout<<"Hessian 矩阵非正定, 求解失败"<<std::endl;
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
        Gui::plotLine(u8"阻尼牛顿法迭代X中间值","x",function.getXStep());
        Gui::wait();

        }

    }
    catch(Exception ex)
    {
        std::cout<<ex.what();
    }
}


