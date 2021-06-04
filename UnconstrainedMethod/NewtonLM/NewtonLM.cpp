#include <Msnhnet/math/MsnhMatrixS.h>
#include <iostream>

using namespace Msnhnet;

class NewtonLM
{
public:
    NewtonLM(int maxIter, double eps, double vk, double rho, double tau):_maxIter(maxIter),_eps(eps),_vk(vk),_rho(rho),_tau(tau){}


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

protected:
    int _maxIter = 100;
    double _eps = 0.00001;
    double _vk  = 3;
    double _rho = 0.2;
    double _tau = 0.9;

protected:
    virtual MatSDS calGradient(const MatSDS& point) = 0;
    virtual MatSDS calHessian(const MatSDS& point) = 0;
    virtual MatSDS calDk(const MatSDS& point) = 0;
    virtual MatSDS function(const MatSDS& point) = 0;
};


class NewtonLMProblem1:public NewtonLM
{
public:
    NewtonLMProblem1(int maxIter, double eps, double vk, double rho, double tau):NewtonLM(maxIter, eps, vk,rho,tau){}

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


    MatSDS calDk(const MatSDS& point) override
    {
        MatSDS J = calGradient(point);
        MatSDS H = calHessian(point);

        MatSDS I = MatSDS::eye(H.mWidth);


        MatSDS Hp  = H + _vk*I;

        if(!isPosMat(Hp))
        {
            H = H + 2*_vk*I;
        }
        else
        {
            H = Hp;
        }

        return -1*H.invert()*J;
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
            //这里就不用检查正定了
            MatSDS dk = calDk(x);

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
    NewtonLMProblem1 function(1000, 0.001,3, 0.4, 0.8);
    MatSDS startPoint(1,2,{0,3});

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
        }

    }
    catch(Exception ex)
    {
        std::cout<<ex.what();
    }
}

