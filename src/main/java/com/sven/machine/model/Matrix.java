package com.sven.machine.model;

public class Matrix<T>
{

    private T x;
    private T y;

    public Matrix(T x, T y)
    {
        this.x = x;
        this.y = y;
    }

    public T getX()
    {
        return x;
    }

    public void setX(T x)
    {
        this.x = x;
    }

    public T getY()
    {
        return y;
    }

    public void setY(T y)
    {
        this.y = y;
    }

}
