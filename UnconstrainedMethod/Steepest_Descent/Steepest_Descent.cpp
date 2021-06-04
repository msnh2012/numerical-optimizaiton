#include <Msnhnet/math/MsnhMatrixS.h>
#include <iostream>

using namespace Msnhnet;

class SteepestDescent
{
public:
    SteepestDescent(double learningRate, int maxIter, double eps):_learningRate(learningRate),_maxIter(maxIter),_eps(eps){}

    void setLearningRate(double learningRate)
    {
        _learningRate = learningRate;
    }

    void setMaxIter(int maxIter)
    {
        _maxIter = maxIter;
    }

    virtual int solve(MatSDS &startPoint) = 0;

    void setEps(double eps)
    {
        _eps = eps;
    }

protected:
    double _learningRate = 0;
    int _maxIter = 100;
    double _eps = 0.00001;

protected:
    virtual MatSDS calGradient(const MatSDS& point) = 0;
    virtual MatSDS function(const MatSDS& point) = 0;
};


class NewtonProblem1:public SteepestDescent
{
public:
    NewtonProblem1(double learningRate, int maxIter, double eps):SteepestDescent(learningRate, maxIter, eps){}

    MatSDS calGradient(const MatSDS &point) override
    {
        MatSDS dk(1,2);
        // df(x) = (2x_1,2x_2)^T
        dk(0,0) = 2*point(0,0);
        dk(0,1) = 2*point(0,1);
        dk = -1*dk;
        return dk;
    }

    MatSDS function(const MatSDS &point) override
    {
        MatSDS f(1,1);
        double x1 = point(0,0);
        double x2 = point(0,1);

        f(0,0) = x1*x1 + x2*x2;

        return f;
    }

    int solve(MatSDS &startPoint) override
    {
        MatSDS x = startPoint;
        for (int i = 0; i < _maxIter; ++i)
        {
            MatSDS dk = calGradient(x);

            if(dk.L2() < _eps)
            {
                startPoint = x;
                return i;
            }
            x = x + _learningRate*dk;
        }

        return -1;
    }
};



int main()
{
    NewtonProblem1 function(0.1, 100, 0.001);
    MatSDS startPoint(1,2,{1,3});
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

