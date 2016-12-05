package com.sven.machine.learning.model;

public class Matrix<T>
{

	public T x;
	public T y;

	public Matrix(T x, T y)
	{
		this.x = x;
		this.y = y;
	}

	@Override
	public String toString()
	{

		return "[" + this.x + "," + this.y + "]";
	}

}
