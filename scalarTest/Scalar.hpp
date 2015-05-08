#ifndef SCALAR_HPP
#define SCALAR_HPP

#include <iostream>
#include <cmath>

namespace cv42 {
	
	template <typename T>
	class Scalar
	{
	protected:
		T			_value;

	public:
		Scalar<T>():_value(0){}
		Scalar<T>(T n):_value(n){}
		~Scalar<T>(){}
		Scalar<T>(Scalar<T> const & rhs){*this = rhs;}

		Scalar<T> & operator=(const Scalar<T> & rhs){
			this->_value = rhs.getValue();
			return *this;
		}
		
		Scalar<T> & operator=(T rhs){
			this->_value = rhs;
			return *this;
		}

		T		getValue(void) const{return this->_value;}
		void	setValue(T const value){_value = value;}

		//prefix operator++
		Scalar<T> & operator++() {
			_value += 1;
			return *this;
		}

		//postfix operator++
		Scalar<T> operator++(int) {
			_value += 1;
			return *this;
		}

		Scalar<T> & operator--() {
			_value -= 1;
			return *this;
		}

		Scalar<T> operator--(int) {
			_value -= 1;
			return *this;
		}

		Scalar<T> & operator+=(const Scalar<T> & rhs){
			_value += rhs.getValue();
			return *this;
		}

		Scalar<T> & operator-=(const Scalar<T> & rhs){
			_value -= rhs.getValue();
			return *this;
		}

		Scalar<T> & operator*=(const Scalar<T> & rhs){
			_value *= rhs.getValue();
			return *this;
		}

		Scalar<T> & operator/=(const Scalar<T> & rhs){
			_value /= rhs.getValue();
			return *this;
		}

		Scalar<T> operator+(const Scalar<T>& rhs) {
			return Scalar<T>(_value + rhs.getValue());
		}

		Scalar<T> operator-(const Scalar<T>& rhs) {
			return Scalar<T>(_value - rhs.getValue());
		}

		const Scalar<T> operator*(const Scalar<T>& rhs) const {
			return Scalar<T>(_value * rhs.getValue());
		}

		Scalar<T> operator*(const Scalar<T>& rhs) {
			return Scalar<T>(_value * rhs.getValue());
		}

		Scalar<T> operator/(const Scalar<T>& rhs) {
			return Scalar<T>(_value / rhs.getValue());
		}

		Scalar<T> operator%(const Scalar<T>& rhs)const {
			return Scalar<T>(static_cast<long>(_value) % static_cast<long>(rhs.getValue()));
		}

		bool operator==(const Scalar<T> &rhs)const {
			return _value == rhs.getValue();
		}
		bool operator!=(const Scalar<T> &rhs)const{
			return _value != rhs.getValue();
		}
		bool operator<(const Scalar<T> &rhs)const{
			return _value < rhs.getValue();
		}
		bool operator>(const Scalar<T> &rhs)const{
			return _value > rhs.getValue();
		}
		bool operator<=(const Scalar<T> &rhs)const{
			return _value <= rhs.getValue();
		}
		bool operator>=(const Scalar<T> &rhs)const{
			return _value >= rhs.getValue();
		}

	};

	template <typename T>
	std::ostream&operator<<(std::ostream & os, Scalar<T> const & rhs)
	{
		os << rhs.getValue();
		return os;
	}

	template <typename T>
	Scalar<T> pow(Scalar<T> &num, Scalar<T> &p) {
		return Scalar<T>(std::pow(num.getValue(), p.getValue()));
	}
}

#endif /*SCALAR_HPP*/
