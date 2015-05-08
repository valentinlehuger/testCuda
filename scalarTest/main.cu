#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <iostream>
#include <vector>


template <typename T>
std::vector<std::vector<T> >				matrix_wise_plus(std::vector<std::vector<T> > mat, T nb)
{
	unsigned int					row_size, col_size;

	col_size = mat.size();
	row_size = mat[0].size();

	thrust::device_vector<float>	d_mat(row_size * col_size);
	thrust::device_vector<float>	d_result(row_size * col_size);


	for (unsigned int i = 0; i < col_size; i += 1)
		thrust::copy(mat[i].begin(), mat[i].end(), d_mat.begin() + (i * row_size));

	thrust::fill(d_result.begin(), d_result.end(), nb);

	thrust::transform(d_mat.begin(), d_mat.end(), d_result.begin(), d_result.begin(), thrust::plus<float>());

	for (unsigned int i = 0; i < col_size; i += 1)
		thrust::copy(d_result.begin() + (i * row_size), d_result.begin() + (i * row_size) + row_size, mat[i].begin());

	return (mat);
}

int								main(void)
{

	std::vector< std::vector<float> >		mat(3, std::vector< float >(3, 1.1));

	for (int i = 0; i < 3; i++)	
	{
		mat[1][i] = 4.4;
		mat[2][i] = 7.7;
	}

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
			std::cout << mat[i][j] << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;
	
	mat = matrix_wise_plus(mat, 1.1f);

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
			std::cout << mat[i][j] << " ";
		std::cout << std::endl;
	}
	return (0);
}




// int main(void)
// {
// 	// allocate three device_vectors with 10 elements




// 	thrust::device_vector<float> X(10);
// 	thrust::device_vector<float> Y(10);
// 	thrust::device_vector<float> Z(10);

// 	// initialize X to 0,1,2,3, ....
// 	// thrust::sequence(X.begin(), X.end());
	
// 	for (float i = 0; i < 10; i++)
// 		X[i] = i;


// 	// // compute Y = -X
// 	// thrust::transform(X.begin(), X.end(), Y.begin(), thrust::negate<int>());

// 	// fill Z with twos
// 	thrust::fill(Z.begin(), Z.end(), 10);
	
// 	// // compute Y = X mod 2
// 	thrust::transform(X.begin(), X.end(), Z.begin(), Y.begin(), thrust::plus<int>());
	
// 	// // replace all the ones in Y with tens
// 	// thrust::replace(Y.begin(), Y.end(), 1, 10);
	
// 	// print Y
// 	// thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<int>(std::cout, "\n"));

// 	Sf a;
// 	for (int i = 0; i < 10; i++)
// 	{
// 		a = Z[i];
// 		std::cout << a << std::endl;
// 	}
// 	return 0;
// }



// #include <thrust/device_vector.h>
// #include <thrust/transform.h>
// #include <thrust/sequence.h>
// #include <thrust/copy.h>
// #include <thrust/fill.h>
// #include <thrust/replace.h>
// #include <thrust/functional.h>
// #include <iostream>
// #include <vector>



// template <typename T>
// struct apx_functor
// {
// 	const T a;
// 	apx_functor(T _a) : a(_a) {}
// 	__host__ __device__
// 		T operator()(const T& x) const {
// 			return a + x;
// 		}
// };

// template <typename T>
// void apx_fast(T A, thrust::device_vector<T>& X, thrust::device_vector<T>& Y)
// {
// 	// Y <- A + X
// 	thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), apx_functor(A));
// 	thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<int>(std::cout, "\n"));
// }



// // square<T> computes the square of a number f(x) -> x*x
// template <typename T>
// struct square
// {
// 	__host__ __device__
// 		T operator()(const T& x) const {
// 			return x * x;
// 		} 
// };


// int main(void)
// {
// 	// initialize host array
// 	std::vector<float> 	x;
// 	std::vector<float>	y;
// 	for (int i = 0; i < 4; i++)
// 		x.push_back(float(i));

// 	// transfer to device
// 	thrust::device_vector<float> d_x(x.begin(), x.end());
// 	thrust::device_vector<float> d_y(x.begin(), x.end());

// 	apx_fast(float(5.), d_x, d_y);
   
	// // setup arguments
	// square<float> unary_op;
	// thrust::plus<float> binary_op;
	// float init = 0;

	// // compute norm
	// float norm = std::sqrt(thrust::transform_reduce(d_x.begin(), d_x.end(), unary_op, init, binary_op));
	
	// std::cout << norm << std::endl;
// 	return 0;
// }
