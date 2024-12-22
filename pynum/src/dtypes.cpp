#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <stdexcept>
#include <cmath>
#include <string>
#include <iomanip>
#include <sstream>
#include <complex>


class int8{
  private:
    int8_t value;

  public:
    explicit int8(int8_t val) : value(val){}
    std::string repr() const { return std::to_string(value); }
    std::string nbits() const { return std::to_string(sizeof(value) * 8); }
    int getValue() const { return static_cast<int>(value); }
    
    int8 operator+(const int8 &other) const { return int8(value + other.value); }
    int8 operator-(const int8 &other) const { return int8(value - other.value); }
    int8 operator*(const int8 &other) const { return int8(value * other.value); }
    int8 floordiv(const int8 &other) const {
      if(other.value == 0) throw std::runtime_error("ZeroDivisionError");
      return int8(value / other.value);
    }
    int8 operator%(const int8 &other) const {
      if(other.value == 0) throw std::runtime_error("ZeroDivisionError");
      return int8(value % other.value);
    }
    int8 power(const int8 &other) const {
      if(other.value < 0) throw std::runtime_error("NegativeExponentError");
      int8_t res = 1;
      for(int8_t i = 0; i < other.value; ++i){ res *= value; }
      return int8(res);
    }
    int8 abs() const {
      if(value < 0) return int8(-value);
      return int8(value);
    }
    int8 neg() const { return int8(-value); }
    int8 pos() const { return int8(value); }

    bool operator==(const int8 &other) const { return value == other.value; }
    bool operator!=(const int8 &other) const { return value != other.value; }
    bool operator>(const int8 &other) const { return value > other.value; }
    bool operator>=(const int8 &other) const { return value >= other.value; }
    bool operator<(const int8 &other) const { return value < other.value; }
    bool operator<=(const int8 &other) const { return value <= other.value; }

    int8 operator<<(const int8 &other) const { return int8(value << other.value); }
    int8 operator>>(const int8 &other) const { return int8(value >> other.value); }

    int8 operator&(const int8 &other) const { return int8(value & other.value); }
    int8 operator|(const int8 &other) const { return int8(value | other.value); }
    int8 operator^(const int8 &other) const { return int8(value ^ other.value); }
    int8 operator~() const { return int8(~value); }
};


class int16{
  private:
    int16_t value;

  public:
    explicit int16(int16_t val) : value(val){}
    std::string repr() const { return std::to_string(value); }
    std::string nbits() const { return std::to_string(sizeof(value) * 8); }
    int getValue() const { return static_cast<int>(value); }
    
    int16 operator+(const int16 &other) const { return int16(value + other.value); }
    int16 operator-(const int16 &other) const { return int16(value - other.value); }
    int16 operator*(const int16 &other) const { return int16(value * other.value); }
    int16 floordiv(const int16 &other) const {
      if(other.value == 0) throw std::runtime_error("ZeroDivisionError");
      return int16(value / other.value);
    }
    int16 operator%(const int16 &other) const {
      if(other.value == 0) throw std::runtime_error("ZeroDivisionError");
      return int16(value % other.value);
    }
    int16 power(const int16 &other) const {
      if(other.value < 0) throw std::runtime_error("NegativeExponentError");
      int16_t res = 1;
      for(int16_t i = 0; i < other.value; ++i){ res *= value; }
      return int16(res);
    }
    int16 abs() const {
      if(value < 0) return int16(-value);
      return int16(value);
    }
    int16 neg() const { return int16(-value); }
    int16 pos() const { return int16(value); }

    bool operator==(const int16 &other) const { return value == other.value; }
    bool operator!=(const int16 &other) const { return value != other.value; }
    bool operator>(const int16 &other) const { return value > other.value; }
    bool operator>=(const int16 &other) const { return value >= other.value; }
    bool operator<(const int16 &other) const { return value < other.value; }
    bool operator<=(const int16 &other) const { return value <= other.value; }

    int16 operator<<(const int16 &other) const { return int16(value << other.value); }
    int16 operator>>(const int16 &other) const { return int16(value >> other.value); }

    int16 operator&(const int16 &other) const { return int16(value & other.value); }
    int16 operator|(const int16 &other) const { return int16(value | other.value); }
    int16 operator^(const int16 &other) const { return int16(value ^ other.value); }
    int16 operator~() const { return int16(~value); }
};


class int32{
  private:
    int32_t value;

  public:
    explicit int32(int32_t val) : value(val){}
    std::string repr() const { return std::to_string(value); }
    std::string nbits() const { return std::to_string(sizeof(value) * 8); }
    int32_t getValue() const { return value; }

    int32 operator+(const int32 &other) const { return int32(value + other.value); }
    int32 operator-(const int32 &other) const { return int32(value - other.value); }
    int32 operator*(const int32 &other) const { return int32(value * other.value); }
    int32 floordiv(const int32 &other) const {
      if(other.value == 0) throw std::runtime_error("ZeroDivisionError");
      return int32(value / other.value);
    }
    int32 operator%(const int32 &other) const {
      if(other.value == 0) throw std::runtime_error("ZeroDivisionError");
      return int32(value % other.value);
    }
    int32 power(const int32 &other) const {
      if(other.value < 0) throw std::runtime_error("NegativeExponentError");
      int32_t res = 1;
      for(int32_t i = 0; i < other.value; ++i){ res *= value; }
      return int32(res);
    }

    int32 abs() const {
      if(value < 0) return int32(-value);
      return int32(value);
    }
    int32 neg() const { return int32(-value); }
    int32 pos() const { return int32(value); }

    bool operator==(const int32 &other) const { return value == other.value; }
    bool operator!=(const int32 &other) const { return value != other.value; }
    bool operator>(const int32 &other) const { return value > other.value; }
    bool operator>=(const int32 &other) const { return value >= other.value; }
    bool operator<(const int32 &other) const { return value < other.value; }
    bool operator<=(const int32 &other) const { return value <= other.value; }

    int32 operator<<(const int32 &other) const { return int32(value << other.value); }
    int32 operator>>(const int32 &other) const { return int32(value >> other.value); }

    int32 operator&(const int32 &other) const { return int32(value & other.value); }
    int32 operator|(const int32 &other) const { return int32(value | other.value); }
    int32 operator^(const int32 &other) const { return int32(value ^ other.value); }
    int32 operator~() const { return int32(~value); }
};


class int64{
  private:
    int64_t value;

  public:
    explicit int64(int64_t val) : value(val){}
    std::string repr() const { return std::to_string(value); }
    std::string nbits() const { return std::to_string(sizeof(value) * 8); }
    int64_t getValue() const { return value; }

    int64 operator+(const int64 &other) const { return int64(value + other.value); }
    int64 operator-(const int64 &other) const { return int64(value - other.value); }
    int64 operator*(const int64 &other) const { return int64(value * other.value); }
    int64 floordiv(const int64 &other) const {
      if(other.value == 0) throw std::runtime_error("ZeroDivisionError");
      return int64(std::floor(value / other.value));
    }
    int64 operator%(const int64 &other) const {
      if(other.value == 0) throw std::runtime_error("ZeroDivisionError");
      return int64(value % other.value);
    }
    int64 power(const int64 &other) const {
      if(other.value == 0) throw std::runtime_error("NegativeExponentError");
      int64_t res = 1;
      for(int64_t i = 0; i < other.value; ++i){ res *= value; }
      return int64(res);
    }

    int64 abs() const {
      if(value < 0) return int64(-value);
      return int64(value);
    }
    int64 neg() const { return int64(-value); }
    int64 pos() const { return int64(value); }

    bool operator==(const int64 &other) const { return value == other.value; }
    bool operator!=(const int64 &other) const { return value != other.value; }
    bool operator>(const int64 &other) const { return value > other.value; }
    bool operator>=(const int64 &other) const { return value >= other.value; }
    bool operator<(const int64 &other) const { return value < other.value; }
    bool operator<=(const int64 &other) const { return value <= other.value; }

    int64 operator<<(const int64 &other) const { return int64(value << other.value); }
    int64 operator>>(const int64 &other) const { return int64(value >> other.value); }

    int64 operator&(const int64 &other) const { return int64(value & other.value); }
    int64 operator|(const int64 &other) const { return int64(value | other.value); }
    int64 operator^(const int64 &other) const { return int64(value ^ other.value); }
    int64 operator~() const { return int64(~value); }
};


class float32{
  private:
    float value;

  public:
    explicit float32(float val) : value(val){}
    std::string repr() const {
      std::ostringstream oss;
      oss << std::setprecision(7) << value;
      return oss.str();
    }
    std::string nbits() const { return std::to_string(sizeof(value) * 8); }
    explicit float32(int16_t val) : value(static_cast<float>(val)){}
    explicit float32(int32_t val) : value(static_cast<float>(val)){}
    explicit float32(int64_t val) : value(static_cast<float>(val)){}
    float getValue() const { return value; }

    float32 operator+(const float32 &other) const { return float32(value + other.value); } 
    float32 operator-(const float32 &other) const { return float32(value - other.value); }
    float32 operator*(const float32 &other) const { return float32(value * other.value); }
    float32 operator/(const float32 &other) const {
      if(other.value == 0) throw std::runtime_error("ZeroDivisionError");
      return float32(value / other.value);
    }
    int32 floordiv(const float32 &other) const { return int32(std::floor(value / other.value)); }
    float32 operator%(const float32 &other) const {
      if(other.value == 0) throw std::runtime_error("ZeroDivisionError");
      return float32(std::fmod(value,other.value));
    }
    float32 power(const float32 &other) const { return float32(std::pow(value,other.value)); }

    float32 abs() const {
      if(value < 0) return float32(-value);
      return float32(value);
    }
    float32 neg() const { return float32(-value); }
    float32 pos() const { return float32(value); }
    
    bool operator==(const float32 &other) const { return value == other.value; }
    bool operator!=(const float32 &other) const { return value != other.value; }
    bool operator>(const float32 &other) const { return value > other.value; }
    bool operator>=(const float32 &other) const { return value >= other.value; }
    bool operator<(const float32 &other) const { return value < other.value; }
    bool operator<=(const float32 &other) const { return value <= other.value; }
};


class float64{
  private:
    double value;

  public:
    explicit float64(double val) : value(val){}
    std::string repr() const {
      std::ostringstream oss;
      oss << std::setprecision(15) << value;
      return oss.str();
    }
    std::string nbits() const { return std::to_string(sizeof(value) * 8); }
    explicit float64(int16_t val) : value(static_cast<double>(val)){}
    explicit float64(int32_t val) : value(static_cast<double>(val)){}
    explicit float64(int64_t val) : value(static_cast<double>(val)){}
    double getValue() const { return value; }
  
    float64 operator+(const float64 &other) const { return float64(value + other.value); }
    float64 operator-(const float64 &other) const { return float64(value - other.value); }
    float64 operator*(const float64 &other) const { return float64(value * other.value); }
    float64 operator/(const float64 &other) const {
      if(other.value == 0) throw std::runtime_error("ZeroDivisionError");
      return float64(value / other.value);
    }
    float64 floordiv(const float64 &other) const {
      if(other.value == 0) throw std::runtime_error("ZeroDivisionError");
      return float64(std::floor(value / other.value));
    }
    float64 operator%(const float64 &other) const {
      if(other.value == 0) throw std::runtime_error("ZeroDivisionError");
      return float64(std::fmod(value,other.value));
    }
    float64 power(const float64 &other) const { return float64(std::pow(value,other.value)); }

    float64 abs() const {
      if(value < 0) return float64(-value);
      return float64(value);
    }
    float64 neg() const { return float64(-value); }
    float64 pos() const { return float64(value); } 

    bool operator==(const float64 &other) const { return value == other.value; }
    bool operator!=(const float64 &other) const { return value != other.value; }
    bool operator>(const float64 &other) const { return value > other.value; }
    bool operator>=(const float64 &other) const { return value >= other.value; }
    bool operator<(const float64 &other) const { return value < other.value; }
    bool operator<=(const float64 &other) const { return value <= other.value; }
};


class complex128{
  private:
    std::complex<double> value;
  
  public:
    explicit complex128(const double &real, const double &imag) : value(real,imag){}
    explicit complex128(const float64 &real, const float64 &imag) : value(real.getValue(),imag.getValue()){}
    explicit complex128(const std::complex<double> &comp) : value(comp){}

    std::string repr() const {
      std::ostringstream oss;
      if(value.imag() < 0) oss << "(" << std::setprecision(15) << value.real() << "-" << -value.imag() << "j)";
      else oss << "(" << std::setprecision(15) << value.real() << "+" << value.imag() << "j)";
      return oss.str();
    }
    std::string nbits() const { return std::to_string(sizeof(value) * 8); }
    std::complex<double> getValue() const { return value; }
    float64 real() const { return float64(value.real()); }
    float64 imag() const { return float64(value.imag()); }

    complex128 operator+(const complex128 &other) const { return complex128(value + other.value); }
    complex128 operator-(const complex128 &other) const { return complex128(value - other.value); }
    complex128 operator*(const complex128 &other) const { return complex128(value * other.value); }
    complex128 operator/(const complex128 &other) const {
      if(other.value == std::complex<double>(0,0)) throw std::runtime_error("ZeroDivisionError");
      return complex128(value / other.value); }
    complex128 conj() const { return complex128(std::conj(value)); }
    float64 phase() const { return float64(std::arg(value)); }
    float64 magnitude() const { return float64(std::abs(value)); }
    complex128 power(const complex128 &other) const { return complex128(std::pow(value,other.value)); }

    bool operator==(const complex128 &other) const { return value == other.value; }
    bool operator!=(const complex128 &other) const { return value != other.value; }
};



// functions for int8 with builtin dtypes
int64 operator+(const int8 &x, const long &y){ return int64(static_cast<long>(x.getValue()) + y); }
int64 operator-(const int8 &x, const long &y){ return int64(static_cast<long>(x.getValue()) - y); }
int64 operator*(const int8 &x, const long &y){ return int64(static_cast<long>(x.getValue()) * y); }
int64 operator/(const int8 &x, const long &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(static_cast<long>(x.getValue()) / y);
}
int64 floordiv(const int8 &x, const long &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(static_cast<long>(x.getValue()) / y);
}
int64 operator%(const int8 &x, const long &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(static_cast<long>(x.getValue()) % y);
}
int64 power(const int8 &x, const long &y){
  if(y < 0) throw std::runtime_error("NegativeExponentError");
  long res = 1;
  for(long i = 0; i < y; ++i){ res *= static_cast<long>(x.getValue()); }
  return int64(res);
}
bool operator==(const int8 &x, const long &y){ return static_cast<long>(x.getValue()) == y; }
bool operator!=(const int8 &x, const long &y){ return static_cast<long>(x.getValue()) != y; }
bool operator>(const int8 &x, const long &y){ return static_cast<long>(x.getValue()) > y; }
bool operator>=(const int8 &x, const long &y){ return static_cast<long>(x.getValue()) >= y; }
bool operator<(const int8 &x, const long &y){ return static_cast<long>(x.getValue()) < y; }
bool operator<=(const int8 &x, const long &y){ return static_cast<long>(x.getValue()) <= y; }
int64 operator<<(const int8 &x, const long &y){ return int64(static_cast<long>(x.getValue()) << y); }
int64 operator>>(const int8 &x, const long &y){ return int64(static_cast<long>(x.getValue()) >> y); }
int64 operator&(const int8 &x, const long &y){ return int64(static_cast<long>(x.getValue()) & y); }
int64 operator|(const int8 &x, const long &y){ return int64(static_cast<long>(x.getValue()) & y); }
int64 operator^(const int8 &x, const long &y){ return int64(static_cast<long>(x.getValue()) & y); }

float64 operator+(const int8 &x, const double &y){ return float64(static_cast<double>(x.getValue()) + y); }
float64 operator-(const int8 &x, const double &y){ return float64(static_cast<double>(x.getValue()) - y); }
float64 operator*(const int8 &x, const double &y){ return float64(static_cast<double>(x.getValue()) * y); }
float64 operator/(const int8 &x, const double &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(static_cast<double>(x.getValue()) / y);
}
float64 floordiv(const int8 &x, const double &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(static_cast<double>(x.getValue()) / y);
}
float64 operator%(const int8 &x, const double &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(fmod(static_cast<double>(x.getValue()),y));
}
float64 power(const int8 &x, const double &y){
  if(y < 0) throw std::runtime_error("NegativeExponentError");
  double res = 1;
  for(double i = 0; i < y; ++i){ res *= static_cast<double>(x.getValue()); }
  return float64(res);
}
bool operator==(const int8 &x, const double &y){ return static_cast<double>(x.getValue()) == y; }
bool operator!=(const int8 &x, const double &y){ return static_cast<double>(x.getValue()) != y; }
bool operator>(const int8 &x, const double &y){ return static_cast<double>(x.getValue()) > y; }
bool operator>=(const int8 &x, const double &y){ return static_cast<double>(x.getValue()) >= y; }
bool operator<(const int8 &x, const double &y){ return static_cast<double>(x.getValue()) < y; }
bool operator<=(const int8 &x, const double &y){ return static_cast<double>(x.getValue()) <= y; }

complex128 operator+(const int8 &x, const std::complex<double> &y){ return complex128(static_cast<double>(x.getValue()) + y.real(),y.imag()); }
complex128 operator-(const int8 &x, const std::complex<double> &y){ return complex128(static_cast<double>(x.getValue()) - y.real(),-y.imag()); }
complex128 operator*(const int8 &x, const std::complex<double> &y){ return complex128(static_cast<double>(x.getValue()) * y.real(),static_cast<double>(x.getValue()) * y.imag()); }
complex128 operator/(const int8 &x, const std::complex<double> &y){
  if(y == std::complex<double>(0,0)) throw std::runtime_error("ZeroDivisionError");
  return complex128(static_cast<double>(x.getValue()) / y.real(),static_cast<double>(x.getValue()) / y.imag());
}
complex128 power(const int8 &x, const std::complex<double> &y){ return complex128(std::pow(static_cast<double>(x.getValue()),y)); }
bool operator==(const int8 &x, const std::complex<double> &y){ return static_cast<double>(x.getValue()) == y.real(); }
bool operator!=(const int8 &x, const std::complex<double> &y){ return static_cast<double>(x.getValue()) != y.real(); }


// function for int8 with int8
float64 operator/(const int8 &x, const int8 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(static_cast<double>(x.getValue()) / static_cast<double>(y.getValue()));
}

// function for int8 with int16
int16 operator+(const int8 &x, const int16 &y){ return int16(static_cast<int16_t>(x.getValue()) + y.getValue()); }
int16 operator-(const int8 &x, const int16 &y){ return int16(static_cast<int16_t>(x.getValue()) - y.getValue()); }
int16 operator*(const int8 &x, const int16 &y){ return int16(static_cast<int16_t>(x.getValue()) * y.getValue()); }
float32 operator/(const int8 &x, const int16 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float32(static_cast<float>(x.getValue()) / static_cast<float>(y.getValue()));
}
int16 floordiv(const int8 &x, const int16 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int16(static_cast<int16_t>(x.getValue()) / y.getValue());
}
int16 operator%(const int8 &x, const int16 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int16(static_cast<int16_t>(x.getValue()) % y.getValue());
}
int16 power(const int8 &x, const int16 &y){
  if(y.getValue() == 0) throw std::runtime_error("NegativeExponentError");
  int16_t res = 1;
  int16_t x_ = static_cast<int16_t>(x.getValue());
  for(int16_t i = 0; i < y.getValue(); ++i){ res *= x_; }
  return int16(res);
}
bool operator==(const int8 &x, const int16 &y){ return static_cast<int16_t>(x.getValue()) == y.getValue(); }
bool operator!=(const int8 &x, const int16 &y){ return static_cast<int16_t>(x.getValue()) != y.getValue(); }
bool operator>(const int8 &x, const int16 &y){ return static_cast<int16_t>(x.getValue()) > y.getValue(); }
bool operator>=(const int8 &x, const int16 &y){ return static_cast<int16_t>(x.getValue()) >= y.getValue(); }
bool operator<(const int8 &x, const int16 &y){ return static_cast<int16_t>(x.getValue()) < y.getValue(); }
bool operator<=(const int8 &x, const int16 &y){ return static_cast<int16_t>(x.getValue()) <= y.getValue(); }
int16 operator<<(const int8 &x, const int16 &y){ return int16(static_cast<int16_t>(x.getValue()) << y.getValue()); }
int16 operator>>(const int8 &x, const int16 &y){ return int16(static_cast<int16_t>(x.getValue()) >> y.getValue()); }
int16 operator&(const int8 &x, const int16 &y){ return int16(static_cast<int16_t>(x.getValue()) & y.getValue()); }
int16 operator^(const int8 &x, const int16 &y){ return int16(static_cast<int16_t>(x.getValue()) ^ y.getValue()); }
int16 operator|(const int8 &x, const int16 &y){ return int16(static_cast<int16_t>(x.getValue()) | y.getValue()); }

// function for int8 with int32
int32 operator+(const int8 &x, const int32 &y){ return int32(static_cast<int32_t>(x.getValue()) + y.getValue()); }
int32 operator-(const int8 &x, const int32 &y){ return int32(static_cast<int32_t>(x.getValue()) - y.getValue()); }
int32 operator*(const int8 &x, const int32 &y){ return int32(static_cast<int32_t>(x.getValue()) * y.getValue()); }
float32 operator/(const int8 &x, const int32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float32(static_cast<float>(x.getValue()) / static_cast<float>(y.getValue()));
}
int32 floordiv(const int8 &x, const int32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int32(static_cast<int32_t>(x.getValue()) / y.getValue());
}
int32 operator%(const int8 &x, const int32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int32(static_cast<int32_t>(x.getValue()) % y.getValue());
}
int32 power(const int8 &x, const int32 &y){
  if(y.getValue() == 0) throw std::runtime_error("NegativeExponentError");
  int32_t res = 1;
  int32_t x_ = static_cast<int32_t>(x.getValue());
  for(int32_t i = 0; i < y.getValue(); ++i){ res *= x_; }
  return int32(res);
}
bool operator==(const int8 &x, const int32 &y){ return static_cast<int32_t>(x.getValue()) == y.getValue(); }
bool operator!=(const int8 &x, const int32 &y){ return static_cast<int32_t>(x.getValue()) != y.getValue(); }
bool operator>(const int8 &x, const int32 &y){ return static_cast<int32_t>(x.getValue()) > y.getValue(); }
bool operator>=(const int8 &x, const int32 &y){ return static_cast<int32_t>(x.getValue()) >= y.getValue(); }
bool operator<(const int8 &x, const int32 &y){ return static_cast<int32_t>(x.getValue()) < y.getValue(); }
bool operator<=(const int8 &x, const int32 &y){ return static_cast<int32_t>(x.getValue()) <= y.getValue(); }
int32 operator<<(const int8 &x, const int32 &y){ return int32(static_cast<int32_t>(x.getValue()) << y.getValue()); }
int32 operator>>(const int8 &x, const int32 &y){ return int32(static_cast<int32_t>(x.getValue()) >> y.getValue()); }
int32 operator&(const int8 &x, const int32 &y){ return int32(static_cast<int32_t>(x.getValue()) & y.getValue()); }
int32 operator^(const int8 &x, const int32 &y){ return int32(static_cast<int32_t>(x.getValue()) ^ y.getValue()); }
int32 operator|(const int8 &x, const int32 &y){ return int32(static_cast<int32_t>(x.getValue()) | y.getValue()); }

//function for int8 with int64
int64 operator+(const int8 &x, const int64 &y){ return int64(static_cast<int64_t>(x.getValue()) + y.getValue()); }
int64 operator-(const int8 &x, const int64 &y){ return int64(static_cast<int64_t>(x.getValue()) - y.getValue()); }
int64 operator*(const int8 &x, const int64 &y){ return int64(static_cast<int64_t>(x.getValue()) * y.getValue()); }
float64 operator/(const int8 &x, const int64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(static_cast<double>(x.getValue()) / static_cast<double>(y.getValue()));
}
int64 floordiv(const int8 &x, const int64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(static_cast<int64_t>(x.getValue()) / y.getValue());
}
int64 operator%(const int8 &x, const int64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(static_cast<int64_t>(x.getValue()) % y.getValue());
}
int64 power(const int8 &x, const int64 &y){
  if(y.getValue() == 0) throw std::runtime_error("NegativeExponentError");
  int64_t res = 1;
  int64_t x_ = static_cast<int64_t>(x.getValue());
  for(int64_t i = 0; i < y.getValue(); ++i){ res *= x_; }
  return int64(res);
}
bool operator==(const int8 &x, const int64 &y){ return static_cast<int64_t>(x.getValue()) == y.getValue(); }
bool operator!=(const int8 &x, const int64 &y){ return static_cast<int64_t>(x.getValue()) != y.getValue(); }
bool operator>(const int8 &x, const int64 &y){ return static_cast<int64_t>(x.getValue()) > y.getValue(); }
bool operator>=(const int8 &x, const int64 &y){ return static_cast<int64_t>(x.getValue()) >= y.getValue(); }
bool operator<(const int8 &x, const int64 &y){ return static_cast<int64_t>(x.getValue()) < y.getValue(); }
bool operator<=(const int8 &x, const int64 &y){ return static_cast<int64_t>(x.getValue()) <= y.getValue(); }
int64 operator<<(const int8 &x, const int64 &y){ return int64(static_cast<int64_t>(x.getValue()) << y.getValue()); }
int64 operator>>(const int8 &x, const int64 &y){ return int64(static_cast<int64_t>(x.getValue()) >> y.getValue()); }
int64 operator&(const int8 &x, const int64 &y){ return int64(static_cast<int64_t>(x.getValue()) & y.getValue()); }
int64 operator^(const int8 &x, const int64 &y){ return int64(static_cast<int64_t>(x.getValue()) ^ y.getValue()); }
int64 operator|(const int8 &x, const int64 &y){ return int64(static_cast<int64_t>(x.getValue()) | y.getValue()); }

// functions for int8 with float32
float32 operator+(const int8 &x, const float32 &y){ return float32(static_cast<float>(x.getValue()) + y.getValue()); }
float32 operator-(const int8 &x, const float32 &y){ return float32(static_cast<float>(x.getValue()) - y.getValue()); }
float32 operator*(const int8 &x, const float32 &y){ return float32(static_cast<float>(x.getValue()) * y.getValue()); }
float32 operator/(const int8 &x, const float32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float32(static_cast<float>(x.getValue()) / y.getValue());
}
int32 floordiv(const int8 &x, const float32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int32(static_cast<int32_t>(x.getValue()) / static_cast<int32_t>(y.getValue()));
}
float32 operator%(const int8 &x, const float32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float32(std::fmod(static_cast<float>(x.getValue()),y.getValue()));
}
float32 power(const int8 &x, const float32 &y){ return float32(std::pow(static_cast<float>(x.getValue()),y.getValue())); }
bool operator==(const int8 &x, const float32 &y){ return static_cast<float>(x.getValue()) == y.getValue(); }
bool operator!=(const int8 &x, const float32 &y){ return static_cast<float>(x.getValue()) != y.getValue(); }
bool operator>(const int8 &x, const float32 &y){ return static_cast<float>(x.getValue()) > y.getValue(); }
bool operator>=(const int8 &x, const float32 &y){ return static_cast<float>(x.getValue()) >= y.getValue(); }
bool operator<(const int8 &x, const float32 &y){ return static_cast<float>(x.getValue()) < y.getValue(); }
bool operator<=(const int8 &x, const float32 &y){ return static_cast<float>(x.getValue()) <= y.getValue(); }

//functions for int8 with float64
float64 operator+(const int8 &x, const float64 &y){ return float64(static_cast<double>(x.getValue()) + y.getValue()); }
float64 operator-(const int8 &x, const float64 &y){ return float64(static_cast<double>(x.getValue()) - y.getValue()); }
float64 operator*(const int8 &x, const float64 &y){ return float64(static_cast<double>(x.getValue()) * y.getValue()); }
float64 operator/(const int8 &x, const float64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(static_cast<double>(x.getValue()) / y.getValue());
}
int64 floordiv(const int8 &x, const float64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(static_cast<int64_t>(x.getValue()) / static_cast<int64_t>(y.getValue()));
}
float64 operator%(const int8 &x, const float64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(std::fmod(static_cast<double>(x.getValue()),y.getValue()));
}
float64 power(const int8 &x, const float64 &y){ return float64(std::pow(static_cast<double>(x.getValue()),y.getValue())); }
bool operator==(const int8 &x, const float64 &y){ return static_cast<double>(x.getValue()) == y.getValue(); }
bool operator!=(const int8 &x, const float64 &y){ return static_cast<double>(x.getValue()) != y.getValue(); }
bool operator>(const int8 &x, const float64 &y){ return static_cast<double>(x.getValue()) > y.getValue(); }
bool operator>=(const int8 &x, const float64 &y){ return static_cast<double>(x.getValue()) >= y.getValue(); }
bool operator<(const int8 &x, const float64 &y){ return static_cast<double>(x.getValue()) < y.getValue(); }
bool operator<=(const int8 &x, const float64 &y){ return static_cast<double>(x.getValue()) <= y.getValue(); }

// functions for int8 with complex128
complex128 operator+(const int8 &x, const complex128 &y){ return complex128(static_cast<double>(x.getValue()) + y.getValue().real(),y.getValue().imag()); }
complex128 operator-(const int8 &x, const complex128 &y){ return complex128(static_cast<double>(x.getValue()) - y.getValue().real(),-y.getValue().imag()); }
complex128 operator*(const int8 &x, const complex128 &y){ return complex128(static_cast<double>(x.getValue()) * y.getValue().real(),static_cast<double>(x.getValue()) * y.getValue().imag()); }
complex128 operator/(const int8 &x, const complex128 &y){
  if(y.getValue() == std::complex<double>(0,0)) throw std::runtime_error("ZeroDivisionError");
  return complex128(static_cast<double>(x.getValue()) / y.getValue().real(),static_cast<double>(x.getValue()) / y.getValue().imag());
}
complex128 power(const int8 &x, const complex128 &y){ return complex128(std::pow(static_cast<double>(x.getValue()),y.getValue())); }
bool operator==(const int8 &x, const complex128 &y){ return static_cast<double>(x.getValue()) == y.getValue().real(); }
bool operator!=(const int8 &x, const complex128 &y){ return static_cast<double>(x.getValue()) != y.getValue().real(); }

// functions for int16 with builtin dtypes
int64 operator+(const int16 &x, const long &y){ return int64(static_cast<long>(x.getValue()) + y); }
int64 operator-(const int16 &x, const long &y){ return int64(static_cast<long>(x.getValue()) - y); }
int64 operator*(const int16 &x, const long &y){ return int64(static_cast<long>(x.getValue()) * y); }
float64 operator/(const int16 &x, const long &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(static_cast<double>(x.getValue()) / static_cast<double>(y));
}
int64 floordiv(const int16 &x, const long &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(static_cast<long>(x.getValue()) / y);
}
int64 operator%(const int16 &x, const long &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(static_cast<long>(x.getValue()) / y);
}
int64 power(const int16 &x, const long &y){
  if(y < 0) throw std::runtime_error("NegativeExponentError");
  long res = 1;
  for(long i = 0; i < y; ++i){ res *= static_cast<long>(x.getValue()); }
  return int64(res);
}
bool operator==(const int16 &x, const long &y){ return static_cast<long>(x.getValue()) == y; }
bool operator!=(const int16 &x, const long &y){ return static_cast<long>(x.getValue()) != y; }
bool operator>(const int16 &x, const long &y){ return static_cast<long>(x.getValue()) > y; }
bool operator>=(const int16 &x, const long &y){ return static_cast<long>(x.getValue()) >= y; }
bool operator<(const int16 &x, const long &y){ return static_cast<long>(x.getValue()) < y; }
bool operator<=(const int16 &x, const long &y){ return static_cast<long>(x.getValue()) <= y; }
int64 operator<<(const int16 &x, const long &y){ return int64(static_cast<int64_t>(x.getValue()) << y); }
int64 operator>>(const int16 &x, const long &y){ return int64(static_cast<int64_t>(x.getValue()) >> y); }
int64 operator&(const int16 &x, const long &y){ return int64(static_cast<int64_t>(x.getValue()) & y); }
int64 operator|(const int16 &x, const long &y){ return int64(static_cast<int64_t>(x.getValue()) | y); }
int64 operator^(const int16 &x, const long &y){ return int64(static_cast<int64_t>(x.getValue()) ^ y); }

float64 operator+(const int16 &x, const double &y){ return float64(static_cast<double>(x.getValue()) + y); }
float64 operator-(const int16 &x, const double &y){ return float64(static_cast<double>(x.getValue()) - y); }
float64 operator*(const int16 &x, const double &y){ return float64(static_cast<double>(x.getValue()) * y); }
float64 operator/(const int16 &x, const double &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(static_cast<double>(x.getValue()) / y);
}
int64 floordiv(const int16 &x, const double &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(std::floor(static_cast<double>(x.getValue()) / y));
}
float64 operator%(const int16 &x, const double &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(std::fmod(static_cast<double>(x.getValue()),y));
}
float64 power(const int16 &x, const double &y){ return float64(std::pow(static_cast<double>(x.getValue()),y)); }
bool operator==(const int16 &x, const double &y){ return static_cast<double>(x.getValue()) == y; }
bool operator!=(const int16 &x, const double &y){ return static_cast<double>(x.getValue()) != y; }
bool operator>(const int16 &x, const double &y){ return static_cast<double>(x.getValue()) > y; }
bool operator>=(const int16 &x, const double &y){ return static_cast<double>(x.getValue()) >= y; }
bool operator<(const int16 &x, const double &y){ return static_cast<double>(x.getValue()) < y; }
bool operator<=(const int16 &x, const double &y){ return static_cast<double>(x.getValue()) <= y; }

complex128 operator+(const int16 &x, const std::complex<double> &y){ return complex128(static_cast<double>(x.getValue()) + y.real(),y.imag()); }
complex128 operator-(const int16 &x, const std::complex<double> &y){ return complex128(static_cast<double>(x.getValue()) - y.real(),-y.imag()); }
complex128 operator*(const int16 &x, const std::complex<double> &y){ return complex128(static_cast<double>(x.getValue()) * y.real(),static_cast<double>(x.getValue()) * y.imag()); }
complex128 operator/(const int16 &x, const std::complex<double> &y){
  if(y == std::complex<double>(0,0)) throw std::runtime_error("ZeroDivisionError");
  return complex128(static_cast<double>(x.getValue()) / y.real(),static_cast<double>(x.getValue()) / y.imag());
}
complex128 power(const int16 &x, const std::complex<double> &y){ return complex128(std::pow(static_cast<double>(x.getValue()),y)); }
bool operator==(const int16 &x, const std::complex<double> &y){ return static_cast<double>(x.getValue()) == y.real(); }
bool operator!=(const int16 &x, const std::complex<double> &y){ return static_cast<double>(x.getValue()) != y.real(); }

// functions for int16 with int16
float32 operator/(const int16 &x, const int16 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float32(static_cast<float>(x.getValue())/static_cast<float>(y.getValue()));
}

// functions for int16 with int8
int16 operator+(const int16 &x, const int8 &y){ return int16(x.getValue() + static_cast<int16_t>(y.getValue())); }
int16 operator-(const int16 &x, const int8 &y){ return int16(x.getValue() - static_cast<int16_t>(y.getValue())); }
int16 operator*(const int16 &x, const int8 &y){ return int16(x.getValue() * static_cast<int16_t>(y.getValue())); }
float32 operator/(const int16 &x, const int8 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float32(static_cast<float>(x.getValue()) / static_cast<float>(y.getValue()));
}
int16 floordiv(const int16 &x, const int8 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int16(x.getValue() / static_cast<int16_t>(y.getValue()));
}
int16 operator%(const int16 &x, const int8 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int16(x.getValue() / static_cast<int16_t>(y.getValue()));
}
int16 power(const int16 &x, const int8 &y){
  if(y.getValue() < 0) throw std::runtime_error("NegativeExponentError");
  int16_t res = 1;
  for(int8_t i = 0; i < y.getValue(); ++i){ res *= x.getValue(); }
  return int16(res);
}
bool operator==(const int16 &x, const int8 &y){ return x.getValue() == static_cast<int16_t>(y.getValue()); }
bool operator!=(const int16 &x, const int8 &y){ return x.getValue() != static_cast<int16_t>(y.getValue()); }
bool operator>(const int16 &x, const int8 &y){ return x.getValue() > static_cast<int16_t>(y.getValue()); }
bool operator>=(const int16 &x, const int8 &y){ return x.getValue() >= static_cast<int16_t>(y.getValue()); }
bool operator<(const int16 &x, const int8 &y){ return x.getValue() < static_cast<int16_t>(y.getValue()); }
bool operator<=(const int16 &x, const int8 &y){ return x.getValue() <= static_cast<int16_t>(y.getValue()); }
int16 operator<<(const int16 &x, const int8 &y){ return int16(x.getValue() << static_cast<int16_t>(y.getValue())); }
int16 operator>>(const int16 &x, const int8 &y){ return int16(x.getValue() >> static_cast<int16_t>(y.getValue())); }
int16 operator&(const int16 &x, const int8 &y){ return int16(x.getValue() & static_cast<int16_t>(y.getValue())); }
int16 operator|(const int16 &x, const int8 &y){ return int16(x.getValue() | static_cast<int16_t>(y.getValue())); }
int16 operator^(const int16 &x, const int8 &y){ return int16(x.getValue() ^ static_cast<int16_t>(y.getValue())); }

// functions for int16 with int32
int32 operator+(const int16 &x, const int32 &y){ return int32(static_cast<int32_t>(x.getValue()) + y.getValue()); }
int32 operator-(const int16 &x, const int32 &y){ return int32(static_cast<int32_t>(x.getValue()) - y.getValue()); }
int32 operator*(const int16 &x, const int32 &y){ return int32(static_cast<int32_t>(x.getValue()) * y.getValue()); }
float32 operator/(const int16 &x, const int32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float32(static_cast<float>(x.getValue()) / static_cast<float>(y.getValue()));
}
int32 floordiv(const int16 &x, const int32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int32(static_cast<int32_t>(x.getValue()) / y.getValue());
}
int32 operator%(const int16 &x, const int32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int32(static_cast<int32_t>(x.getValue()) / y.getValue());
}
int32 power(const int16 &x, const int32 &y){
  if(y.getValue() < 0) throw std::runtime_error("NegativeExponentError");
  int32_t res = 1;
  for(int32_t i = 0; i < y.getValue(); ++i){ res *= x.getValue(); }
  return int32(res);
}
bool operator==(const int16 &x, const int32 &y){ return static_cast<int32_t>(x.getValue()) == y.getValue(); }
bool operator!=(const int16 &x, const int32 &y){ return static_cast<int32_t>(x.getValue()) != y.getValue(); }
bool operator>(const int16 &x, const int32 &y){ return static_cast<int32_t>(x.getValue()) > y.getValue(); }
bool operator>=(const int16 &x, const int32 &y){ return static_cast<int32_t>(x.getValue()) >= y.getValue(); }
bool operator<(const int16 &x, const int32 &y){ return static_cast<int32_t>(x.getValue()) < y.getValue(); }
bool operator<=(const int16 &x, const int32 &y){ return static_cast<int32_t>(x.getValue()) <= y.getValue(); }
int32 operator<<(const int16 &x, const int32 &y){ return int32(static_cast<int32_t>(x.getValue()) << y.getValue()); }
int32 operator>>(const int16 &x, const int32 &y){ return int32(static_cast<int32_t>(x.getValue()) >> y.getValue()); }
int32 operator&(const int16 &x, const int32 &y){ return int32(static_cast<int32_t>(x.getValue()) & y.getValue()); }
int32 operator|(const int16 &x, const int32 &y){ return int32(static_cast<int32_t>(x.getValue()) | y.getValue()); }
int32 operator^(const int16 &x, const int32 &y){ return int32(static_cast<int32_t>(x.getValue()) ^ y.getValue()); }

// functions for int16 with int64
int64 operator+(const int16 &x, const int64 &y){ return int64(static_cast<int64_t>(x.getValue()) + y.getValue()); }
int64 operator-(const int16 &x, const int64 &y){ return int64(static_cast<int64_t>(x.getValue()) - y.getValue()); }
int64 operator*(const int16 &x, const int64 &y){ return int64(static_cast<int64_t>(x.getValue()) * y.getValue()); }
float64 operator/(const int16 &x, const int64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(static_cast<double>(x.getValue()) / static_cast<double>(y.getValue()));
}
int64 floordiv(const int16 &x, const int64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(static_cast<int64_t>(x.getValue()) / y.getValue());
}
int64 operator%(const int16 &x, const int64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(static_cast<int64_t>(x.getValue()) % y.getValue());
}
int64 power(const int16 &x, const int64 &y){
  if(y.getValue() < 0) throw std::runtime_error("NegativeExponentError");
  int64_t res = 1;
  for(int64_t i = 0; i < y.getValue(); ++i){ res *= static_cast<int64_t>(x.getValue()); }
  return int64(res);
}
bool operator==(const int16 &x, const int64 &y){ return static_cast<int64_t>(x.getValue()) == y.getValue(); }
bool operator!=(const int16 &x, const int64 &y){ return static_cast<int64_t>(x.getValue()) != y.getValue(); }
bool operator>(const int16 &x, const int64 &y){ return static_cast<int64_t>(x.getValue()) > y.getValue(); }
bool operator>=(const int16 &x, const int64 &y){ return static_cast<int64_t>(x.getValue()) >= y.getValue(); }
bool operator<(const int16 &x, const int64 &y){ return static_cast<int64_t>(x.getValue()) < y.getValue(); }
bool operator<=(const int16 &x, const int64 &y){ return static_cast<int64_t>(x.getValue()) <= y.getValue(); }
int64 operator<<(const int16 &x, const int64 &y){ return int64(static_cast<int64_t>(x.getValue()) << y.getValue()); }
int64 operator>>(const int16 &x, const int64 &y){ return int64(static_cast<int64_t>(x.getValue()) >> y.getValue()); }
int64 operator&(const int16 &x, const int64 &y){ return int64(static_cast<int64_t>(x.getValue()) & y.getValue()); }
int64 operator|(const int16 &x, const int64 &y){ return int64(static_cast<int64_t>(x.getValue()) | y.getValue()); }
int64 operator^(const int16 &x, const int64 &y){ return int64(static_cast<int64_t>(x.getValue()) ^ y.getValue()); }

// functions for int16 with float32
float32 operator+(const int16 &x, const float32 &y){ return float32(static_cast<float>(x.getValue()) + y.getValue()); }
float32 operator-(const int16 &x, const float32 &y){ return float32(static_cast<float>(x.getValue()) - y.getValue()); }
float32 operator*(const int16 &x, const float32 &y){ return float32(static_cast<float>(x.getValue()) * y.getValue()); }
float32 operator/(const int16 &x, const float32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float32(static_cast<float>(x.getValue()) / y.getValue());
}
float32 floordiv(const int16 &x, const float32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float32(std::floor(static_cast<float>(x.getValue()) / y.getValue()));
}
float32 operator%(const int16 &x, const float32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float32(std::fmod(static_cast<float>(x.getValue()),y.getValue()));
}
float32 power(const int16 &x, const float32 &y){
  if(y.getValue() < 0) throw std::runtime_error("NegativeExponentError");
  return float32(std::pow(static_cast<float>(x.getValue()),y.getValue()));
}
bool operator==(const int16 &x, const float32 &y){ return static_cast<float>(x.getValue()) == y.getValue(); }
bool operator!=(const int16 &x, const float32 &y){ return static_cast<float>(x.getValue()) != y.getValue(); }
bool operator>(const int16 &x, const float32 &y){ return static_cast<float>(x.getValue()) > y.getValue(); }
bool operator>=(const int16 &x, const float32 &y){ return static_cast<float>(x.getValue()) >= y.getValue(); }
bool operator<(const int16 &x, const float32 &y){ return static_cast<float>(x.getValue()) < y.getValue(); }
bool operator<=(const int16 &x, const float32 &y){ return static_cast<float>(x.getValue()) <= y.getValue(); }

// functions for int16 with float64
float64 operator+(const int16 &x, const float64 &y){ return float64(static_cast<double>(x.getValue()) + y.getValue()); }
float64 operator-(const int16 &x, const float64 &y){ return float64(static_cast<double>(x.getValue()) - y.getValue()); }
float64 operator*(const int16 &x, const float64 &y){ return float64(static_cast<double>(x.getValue()) * y.getValue()); }
float64 operator/(const int16 &x, const float64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(static_cast<float>(x.getValue()) / y.getValue());
}
float64 floordiv(const int16 &x, const float64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(std::floor(static_cast<double>(x.getValue()) / y.getValue()));
}
float64 operator%(const int16 &x, const float64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(std::fmod(static_cast<double>(x.getValue()),y.getValue()));
}
float64 power(const int16 &x, const float64 &y){
  if(y.getValue() < 0) throw std::runtime_error("NegativeExponentError");
  return float64(std::pow(static_cast<double>(x.getValue()),y.getValue()));
}
bool operator==(const int16 &x, const float64 &y){ return static_cast<double>(x.getValue()) == y.getValue(); }
bool operator!=(const int16 &x, const float64 &y){ return static_cast<double>(x.getValue()) != y.getValue(); }
bool operator>(const int16 &x, const float64 &y){ return static_cast<double>(x.getValue()) > y.getValue(); }
bool operator>=(const int16 &x, const float64 &y){ return static_cast<double>(x.getValue()) >= y.getValue(); }
bool operator<(const int16 &x, const float64 &y){ return static_cast<double>(x.getValue()) < y.getValue(); }
bool operator<=(const int16 &x, const float64 &y){ return static_cast<double>(x.getValue()) <= y.getValue(); }

// functions for int16 with complex128
complex128 operator+(const int16 &x, const complex128 &y){ return complex128(static_cast<double>(x.getValue()) + y.getValue().real(),y.getValue().imag()); }
complex128 operator-(const int16 &x, const complex128 &y){ return complex128(static_cast<double>(x.getValue()) - y.getValue().real(),-y.getValue().imag()); }
complex128 operator*(const int16 &x, const complex128 &y){ return complex128(static_cast<double>(x.getValue()) * y.getValue().real(),static_cast<double>(x.getValue()) * y.getValue().imag()); }
complex128 operator/(const int16 &x, const complex128 &y){
  if(y.getValue() == std::complex<double>(0,0)) throw std::runtime_error("ZeroDivisionError");
  return complex128(static_cast<double>(x.getValue()) / y.getValue().real(),static_cast<double>(x.getValue()) / y.getValue().imag());
}
complex128 power(const int16 &x, const complex128 &y){ return complex128(std::pow(static_cast<double>(x.getValue()),y.getValue())); }
bool operator==(const int16 &x, const complex128 &y){ return static_cast<double>(x.getValue()) == y.getValue().real(); }
bool operator!=(const int16 &x, const complex128 &y){ return static_cast<double>(x.getValue()) != y.getValue().real(); }

// functions for int32 with builtin dtypes
int64 operator+(const int32 &x, const long &y){ return int64(static_cast<long>(x.getValue()) + y); }
int64 operator-(const int32 &x, const long &y){ return int64(static_cast<long>(x.getValue()) - y); }
int64 operator*(const int32 &x, const long &y){ return int64(static_cast<long>(x.getValue()) * y); }
float64 operator/(const int32 &x, const long &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(static_cast<double>(x.getValue()) / static_cast<double>(y));
}
int64 floordiv(const int32 &x, const long &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(static_cast<long>(x.getValue()) / static_cast<long>(y)); }
int64 operator%(const int32 &x, const long &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(static_cast<long>(x.getValue()) % static_cast<long>(y)); }
int64 power(const int32 &x, const long &y){
  if(y < 0) throw std::runtime_error("NegativeExponentError");
  long res = 1;
  for(long i = 0; i < y; ++i){ res *= x.getValue(); }
  return int64(res);
}
bool operator==(const int32 &x, const long &y){ return x.getValue() == static_cast<long>(y); }
bool operator!=(const int32 &x, const long &y){ return x.getValue() != static_cast<long>(y); }
bool operator>(const int32 &x, const long &y){ return x.getValue() > static_cast<long>(y); }
bool operator>=(const int32 &x, const long &y){ return x.getValue() >= static_cast<long>(y); }
bool operator<(const int32 &x, const long &y){ return x.getValue() < static_cast<long>(y); }
bool operator<=(const int32 &x, const long &y){ return x.getValue() <= static_cast<long>(y); }
int64 operator<<(const int32 &x, const long &y){ return int64(static_cast<int64_t>(x.getValue()) << y); }
int64 operator>>(const int32 &x, const long &y){ return int64(static_cast<int64_t>(x.getValue()) >> y); }
int64 operator&(const int32 &x, const long &y){ return int64(static_cast<int64_t>(x.getValue()) & y); }
int64 operator|(const int32 &x, const long &y){ return int64(static_cast<int64_t>(x.getValue()) | y); }
int64 operator^(const int32 &x, const long &y){ return int64(static_cast<int64_t>(x.getValue()) ^ y); }

float64 operator+(const int32 &x, const double &y){ return float64(static_cast<double>(x.getValue() + y)); }
float64 operator-(const int32 &x, const double &y){ return float64(static_cast<double>(x.getValue() - y)); }
float64 operator*(const int32 &x, const double &y){ return float64(static_cast<double>(x.getValue() * y)); }
float64 operator/(const int32 &x, const double &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(static_cast<double>(x.getValue() / y));
}
int64 floordiv(const int32 &x, const double &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(x.getValue() / static_cast<int32_t>(y));
}
float64 operator%(const int32 &x, const double &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(std::fmod(static_cast<double>(x.getValue()),y));
}
float64 power(const int32 &x, const double &y){ return float64(std::pow(static_cast<double>(x.getValue()),y)); }
bool operator==(const int32 &x, const double &y){ return static_cast<double>(x.getValue()) == y; }
bool operator!=(const int32 &x, const double &y){ return static_cast<double>(x.getValue()) != y; }
bool operator>(const int32 &x, const double &y){ return static_cast<double>(x.getValue()) > y; }
bool operator>=(const int32 &x, const double &y){ return static_cast<double>(x.getValue()) >= y; }
bool operator<(const int32 &x, const double &y){ return static_cast<double>(x.getValue()) < y; }
bool operator<=(const int32 &x, const double &y){ return static_cast<double>(x.getValue()) <= y; }

complex128 operator+(const int32 &x, const std::complex<double> &y){ return complex128(static_cast<double>(x.getValue()) + y.real(),y.imag()); }
complex128 operator-(const int32 &x, const std::complex<double> &y){ return complex128(static_cast<double>(x.getValue()) - y.real(),-y.imag()); }
complex128 operator*(const int32 &x, const std::complex<double> &y){ return complex128(static_cast<double>(x.getValue()) * y.real(),static_cast<double>(x.getValue()) * y.imag()); }
complex128 operator/(const int32 &x, const std::complex<double> &y){
  if(y == std::complex<double>(0,0)) throw std::runtime_error("ZeroDivisionError");
  return complex128(static_cast<double>(x.getValue()) / y.real(),static_cast<double>(x.getValue()) / y.imag());
}
complex128 power(const int32 &x, const std::complex<double> &y){ return complex128(std::pow(static_cast<double>(x.getValue()),y)); }
bool operator==(const int32 &x, const std::complex<double> &y){ return static_cast<double>(x.getValue()) == y.real(); }
bool operator!=(const int32 &x, const std::complex<double> &y){ return static_cast<double>(x.getValue()) != y.real(); }

// functions for int32 with int32
float32 operator/(const int32 &x, const int32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float32(static_cast<float>(x.getValue()) / static_cast<float>(y.getValue()));
}

// functions for int32 with int8
int32 operator+(const int32 &x, const int8 &y){ return int32(x.getValue() + static_cast<int32_t>(y.getValue())); }
int32 operator-(const int32 &x, const int8 &y){ return int32(x.getValue() + static_cast<int32_t>(y.getValue())); }
int32 operator*(const int32 &x, const int8 &y){ return int32(x.getValue() + static_cast<int32_t>(y.getValue())); }
float32 operator/(const int32 &x, const int8 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float32(static_cast<float>(x.getValue()) / static_cast<float>(y.getValue()));
}
int32 floordiv(const int32 &x, const int8 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int32(x.getValue() / static_cast<int32_t>(y.getValue()));
}
int32 operator%(const int32 &x, const int8 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int32(x.getValue() % static_cast<int32_t>(y.getValue()));
}
int32 power(const int32 &x, const int8 &y){
  if(y.getValue() < 0) throw std::runtime_error("NegativeExponentError");
  int16_t res = static_cast<int16_t>(y.getValue());
  for(int16_t i = 0; i < static_cast<int16_t>(y.getValue()); ++i){ res *= x.getValue(); }
  return int32(res);
}
bool operator==(const int32 &x, const int8 &y){ return x.getValue() == static_cast<int32_t>(y.getValue()); }
bool operator!=(const int32 &x, const int8 &y){ return x.getValue() != static_cast<int32_t>(y.getValue()); }
bool operator>(const int32 &x, const int8 &y){ return x.getValue() > static_cast<int32_t>(y.getValue()); }
bool operator>=(const int32 &x, const int8 &y){ return x.getValue() >= static_cast<int32_t>(y.getValue()); }
bool operator<(const int32 &x, const int8 &y){ return x.getValue() < static_cast<int32_t>(y.getValue()); }
bool operator<=(const int32 &x, const int8 &y){ return x.getValue() <= static_cast<int32_t>(y.getValue()); }
int32 operator<<(const int32 &x, const int8 &y){ return int32(x.getValue() << static_cast<int32_t>(y.getValue())); }
int32 operator>>(const int32 &x, const int8 &y){ return int32(x.getValue() >> static_cast<int32_t>(y.getValue())); }
int32 operator&(const int32 &x, const int8 &y){ return int32(x.getValue() & static_cast<int32_t>(y.getValue())); }
int32 operator|(const int32 &x, const int8 &y){ return int32(x.getValue() | static_cast<int32_t>(y.getValue())); }
int32 operator^(const int32 &x, const int8 &y){ return int32(x.getValue() ^ static_cast<int32_t>(y.getValue())); }

// functions for int32 with int16
int32 operator+(const int32 &x, const int16 &y){ return int32(x.getValue() + static_cast<int32_t>(y.getValue())); }
int32 operator-(const int32 &x, const int16 &y){ return int32(x.getValue() + static_cast<int32_t>(y.getValue())); }
int32 operator*(const int32 &x, const int16 &y){ return int32(x.getValue() + static_cast<int32_t>(y.getValue())); }
float32 operator/(const int32 &x, const int16 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float32(static_cast<float>(x.getValue()) / static_cast<float>(y.getValue()));
}
int32 floordiv(const int32 &x, const int16 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int32(x.getValue() / static_cast<int32_t>(y.getValue()));
}
int32 operator%(const int32 &x, const int16 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int32(x.getValue() % static_cast<int32_t>(y.getValue()));
}
int32 power(const int32 &x, const int16 &y){
  if(y.getValue() < 0) throw std::runtime_error("NegativeExponentError");
  int16_t res = static_cast<int16_t>(y.getValue());
  for(int16_t i = 0; i < static_cast<int16_t>(y.getValue()); ++i){ res *= x.getValue(); }
  return int32(res);
}
bool operator==(const int32 &x, const int16 &y){ return x.getValue() == static_cast<int32_t>(y.getValue()); }
bool operator!=(const int32 &x, const int16 &y){ return x.getValue() != static_cast<int32_t>(y.getValue()); }
bool operator>(const int32 &x, const int16 &y){ return x.getValue() > static_cast<int32_t>(y.getValue()); }
bool operator>=(const int32 &x, const int16 &y){ return x.getValue() >= static_cast<int32_t>(y.getValue()); }
bool operator<(const int32 &x, const int16 &y){ return x.getValue() < static_cast<int32_t>(y.getValue()); }
bool operator<=(const int32 &x, const int16 &y){ return x.getValue() <= static_cast<int32_t>(y.getValue()); }
int32 operator<<(const int32 &x, const int16 &y){ return int32(x.getValue() << static_cast<int32_t>(y.getValue())); }
int32 operator>>(const int32 &x, const int16 &y){ return int32(x.getValue() >> static_cast<int32_t>(y.getValue())); }
int32 operator&(const int32 &x, const int16 &y){ return int32(x.getValue() & static_cast<int32_t>(y.getValue())); }
int32 operator|(const int32 &x, const int16 &y){ return int32(x.getValue() | static_cast<int32_t>(y.getValue())); }
int32 operator^(const int32 &x, const int16 &y){ return int32(x.getValue() ^ static_cast<int32_t>(y.getValue())); }

// functions for int32 with int64
int64 operator+(const int32 &x, const int64 &y){ return int64(static_cast<int64_t>(x.getValue()) + y.getValue()); }
int64 operator-(const int32 &x, const int64 &y){ return int64(static_cast<int64_t>(x.getValue()) + y.getValue()); }
int64 operator*(const int32 &x, const int64 &y){ return int64(static_cast<int64_t>(x.getValue()) + y.getValue()); }
float64 operator/(const int32 &x, const int64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(static_cast<double>(x.getValue()) / static_cast<double>(y.getValue()));
}
int64 floordiv(const int32 &x, const int64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(x.getValue() / static_cast<int32_t>(y.getValue()));
}
int64 operator%(const int32 &x, const int64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(x.getValue() % static_cast<int64_t>(y.getValue()));
}
int64 power(const int32 &x, const int64 &y){
  if(y.getValue() < 0) throw std::runtime_error("NegativeExponentError");
  int64_t res = static_cast<int64_t>(y.getValue());
  for(int64_t i = 0; i < static_cast<int64_t>(y.getValue()); ++i){ res *= x.getValue(); }
  return int64(res);
}
bool operator==(const int32 &x, const int64 &y){ return static_cast<int64_t>(x.getValue()) == y.getValue(); }
bool operator!=(const int32 &x, const int64 &y){ return static_cast<int64_t>(x.getValue()) != y.getValue(); }
bool operator>(const int32 &x, const int64 &y){ return static_cast<int64_t>(x.getValue()) > y.getValue(); }
bool operator>=(const int32 &x, const int64 &y){ return static_cast<int64_t>(x.getValue()) >= y.getValue(); }
bool operator<(const int32 &x, const int64 &y){ return static_cast<int64_t>(x.getValue()) < y.getValue(); }
bool operator<=(const int32 &x, const int64 &y){ return static_cast<int64_t>(x.getValue()) <= y.getValue(); }
int64 operator<<(const int32 &x, const int64 &y){ return int64(static_cast<int64_t>(x.getValue()) << y.getValue()); }
int64 operator>>(const int32 &x, const int64 &y){ return int64(static_cast<int64_t>(x.getValue()) >> y.getValue()); }
int64 operator&(const int32 &x, const int64 &y){ return int64(static_cast<int64_t>(x.getValue()) & y.getValue()); }
int64 operator|(const int32 &x, const int64 &y){ return int64(static_cast<int64_t>(x.getValue()) | y.getValue()); }
int64 operator^(const int32 &x, const int64 &y){ return int64(static_cast<int64_t>(x.getValue()) ^ y.getValue()); }

// functions for int32 with float32
float32 operator+(const int32 &x, const float32 &y){ return float32(static_cast<float>(x.getValue()) + y.getValue()); }
float32 operator-(const int32 &x, const float32 &y){ return float32(static_cast<float>(x.getValue()) - y.getValue()); }
float32 operator*(const int32 &x, const float32 &y){ return float32(static_cast<float>(x.getValue()) * y.getValue()); }
float32 operator/(const int32 &x, const float32 &y){
  if(y.getValue() == 0)  throw std::runtime_error("ZeroDivisionError");
  return float32(static_cast<float>(x.getValue()) / y.getValue()); }
int32 floordiv(const int32 &x, const float32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int32(std::floor(static_cast<float>(x.getValue()) / y.getValue()));
}
float32 operator%(const int32 &x, const float32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float32(std::fmod(static_cast<float>(x.getValue()),y.getValue()));
}
float32 power(const int32 &x, const float32 &y){ return float32(std::pow(static_cast<float>(x.getValue()),y.getValue())); }
bool operator==(const int32 &x, const float32 &y){ return static_cast<float>(x.getValue()) == y.getValue(); }
bool operator!=(const int32 &x, const float32 &y){ return static_cast<float>(x.getValue()) != y.getValue(); }
bool operator>(const int32 &x, const float32 &y){ return static_cast<float>(x.getValue()) < y.getValue(); }
bool operator>=(const int32 &x, const float32 &y){ return static_cast<float>(x.getValue()) >= y.getValue(); }
bool operator<(const int32 &x, const float32 &y){ return static_cast<float>(x.getValue()) < y.getValue(); }
bool operator<=(const int32 &x, const float32 &y){ return static_cast<float>(x.getValue()) <= y.getValue(); }

// functions for int32 with float64
float64 operator+(const int32 &x, const float64 &y){ return float64(static_cast<double>(x.getValue()) + y.getValue()); }
float64 operator-(const int32 &x, const float64 &y){ return float64(static_cast<double>(x.getValue()) - y.getValue()); }
float64 operator*(const int32 &x, const float64 &y){ return float64(static_cast<double>(x.getValue()) * y.getValue()); }
float64 operator/(const int32 &x, const float64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(static_cast<double>(x.getValue()) / y.getValue()); }
int64 floordiv(const int32 &x, const float64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(std::floor(static_cast<double>(x.getValue()) / y.getValue()));
}
float64 operator%(const int32 &x, const float64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(std::fmod(static_cast<double>(x.getValue()),y.getValue()));
}
float64 power(const int32 &x, const float64 &y){ return float64(std::pow(static_cast<double>(x.getValue()),y.getValue())); }
bool operator==(const int32 &x, const float64 &y){ return static_cast<double>(x.getValue()) == y.getValue(); }
bool operator!=(const int32 &x, const float64 &y){ return static_cast<double>(x.getValue()) != y.getValue(); }
bool operator>(const int32 &x, const float64 &y){ return static_cast<double>(x.getValue()) < y.getValue(); }
bool operator>=(const int32 &x, const float64 &y){ return static_cast<double>(x.getValue()) >= y.getValue(); }
bool operator<(const int32 &x, const float64 &y){ return static_cast<double>(x.getValue()) < y.getValue(); }
bool operator<=(const int32 &x, const float64 &y){ return static_cast<double>(x.getValue()) <= y.getValue(); }

// functions for int32 with complex128
complex128 operator+(const int32 &x, const complex128 &y){ return complex128(static_cast<double>(x.getValue()) + y.getValue().real(),y.getValue().imag()); }
complex128 operator-(const int32 &x, const complex128 &y){ return complex128(static_cast<double>(x.getValue()) - y.getValue().real(),-y.getValue().imag()); }
complex128 operator*(const int32 &x, const complex128 &y){ return complex128(static_cast<double>(x.getValue()) * y.getValue().real(),static_cast<double>(x.getValue()) * y.getValue().imag()); }
complex128 operator/(const int32 &x, const complex128 &y){
  if(y.getValue() == std::complex<double>(0,0)) throw std::runtime_error("ZeroDivisionError");
  return complex128(static_cast<double>(x.getValue()) / y.getValue().real(),static_cast<double>(x.getValue()) / y.getValue().imag());
}
complex128 power(const int32 &x, const complex128 &y){ return complex128(std::pow(static_cast<double>(x.getValue()),y.getValue())); }
bool operator==(const int32 &x, const complex128 &y){ return static_cast<double>(x.getValue()) == y.getValue().real(); }
bool operator!=(const int32 &x, const complex128 &y){ return static_cast<double>(x.getValue()) != y.getValue().real(); }

// functions for int64 with builtin dtypes
int64 operator+(const int64 &x, const long &y){ return int64(x.getValue() + y); }
int64 operator-(const int64 &x, const long &y){ return int64(x.getValue() - y); }
int64 operator*(const int64 &x, const long &y){ return int64(x.getValue() * y); }
float64 operator/(const int64 &x, const long &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(static_cast<double>(x.getValue()) / static_cast<double>(y));
}
int64 floordiv(const int64 &x, const long &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(x.getValue() / y);
}
int64 operator%(const int64 &x, const long &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(x.getValue() % y);
}
int64 power(const int64 &x, const long &y){
  if(y < 0) throw std::runtime_error("NegativeExponentError");
  int64_t res = 1;
  for(int64_t i = 0; i < y; ++i){ res *= x.getValue(); }
  return int64(res);
}
bool operator==(const int64 &x, const long &y){ return x.getValue() == y; }
bool operator!=(const int64 &x, const long &y){ return x.getValue() != y; }
bool operator>(const int64 &x, const long &y){ return x.getValue() > y; }
bool operator>=(const int64 &x, const long &y){ return x.getValue() >= y; }
bool operator<(const int64 &x, const long &y){ return x.getValue() < y; }
bool operator<=(const int64 &x, const long &y){ return x.getValue() <= y; }
int64 operator<<(const int64 &x, const long &y){ return int64(x.getValue() << y); }
int64 operator>>(const int64 &x, const long &y){ return int64(x.getValue() >> y); }
int64 operator&(const int64 &x, const long &y){ return int64(x.getValue() & y); }
int64 operator|(const int64 &x, const long &y){ return int64(x.getValue() | y); }
int64 operator^(const int64 &x, const long &y){ return int64(x.getValue() ^ y); }

float64 operator+(const int64 &x, const double &y){ return float64(static_cast<double>(x.getValue()) + y); }
float64 operator-(const int64 &x, const double &y){ return float64(static_cast<double>(x.getValue()) - y); }
float64 operator*(const int64 &x, const double &y){ return float64(static_cast<double>(x.getValue()) * y); }
float64 operator/(const int64 &x, const double &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(static_cast<double>(x.getValue()) / static_cast<double>(y));
}
float64 floordiv(const int64 &x, const double &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(static_cast<double>(x.getValue()) / y);
}
float64 operator%(const int64 &x, const double &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(std::fmod(static_cast<double>(x.getValue()),y));
}
float64 power(const int64 &x, const double &y){ return float64(std::pow(static_cast<double>(x.getValue()),y)); }
bool operator==(const int64 &x, const double &y){ return static_cast<double>(x.getValue()) == y; }
bool operator!=(const int64 &x, const double &y){ return static_cast<double>(x.getValue()) != y; }
bool operator>(const int64 &x, const double &y){ return static_cast<double>(x.getValue()) > y; }
bool operator>=(const int64 &x, const double &y){ return static_cast<double>(x.getValue()) >= y; }
bool operator<(const int64 &x, const double &y){ return static_cast<double>(x.getValue()) < y; }
bool operator<=(const int64 &x, const double &y){ return static_cast<double>(x.getValue()) <= y; }

complex128 operator+(const int64 &x, const std::complex<double> &y){ return complex128(static_cast<double>(x.getValue()) + y.real(),y.imag()); }
complex128 operator-(const int64 &x, const std::complex<double> &y){ return complex128(static_cast<double>(x.getValue()) - y.real(),-y.imag()); }
complex128 operator*(const int64 &x, const std::complex<double> &y){ return complex128(static_cast<double>(x.getValue()) * y.real(),static_cast<double>(x.getValue()) * y.imag()); }
complex128 operator/(const int64 &x, const std::complex<double> &y){
  if(y == std::complex<double>(0,0)) throw std::runtime_error("ZeroDivisionError");
  return complex128(static_cast<double>(x.getValue()) / y.real(),static_cast<double>(x.getValue()) / y.imag());
}
complex128 power(const int64 &x, const std::complex<double> &y){ return complex128(std::pow(static_cast<double>(x.getValue()),y)); }
bool operator==(const int64 &x, const std::complex<double> &y){ return static_cast<double>(x.getValue()) == y.real(); }
bool operator!=(const int64 &x, const std::complex<double> &y){ return static_cast<double>(x.getValue()) != y.real(); }

// functions for int64 with int64
float64 operator/(const int64 &x, const int64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(static_cast<double>(x.getValue()) / static_cast<double>(y.getValue()));
}

// functions for int64 with int8
int64 operator+(const int64 &x, const int8 &y){ return int64(x.getValue() + static_cast<int64_t>(y.getValue())); }
int64 operator-(const int64 &x, const int8 &y){ return int64(x.getValue() - static_cast<int64_t>(y.getValue())); }
int64 operator*(const int64 &x, const int8 &y){ return int64(x.getValue() * static_cast<int64_t>(y.getValue())); }
float64 operator/(const int64 &x, const int8 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(static_cast<double>(x.getValue()) / static_cast<double>(y.getValue()));
}
int64 floordiv(const int64 &x, const int8 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(x.getValue() + static_cast<int64_t>(y.getValue()));
}
int64 operator%(const int64 &x, const int8 &y){ return int64(x.getValue() % static_cast<int64_t>(y.getValue())); }
int64 power(const int64 &x, const int8 &y){
  if(y.getValue() < 0) throw std::runtime_error("NegativeExponentError");
  int64_t res = 1;
  for(int8_t i = 0; i < y.getValue(); ++i){ res *= x.getValue(); }
  return int64(res);
}
bool operator==(const int64 &x, const int8 &y){ return x.getValue() == static_cast<int64_t>(y.getValue()); }
bool operator!=(const int64 &x, const int8 &y){ return x.getValue() != static_cast<int64_t>(y.getValue()); }
bool operator>(const int64 &x, const int8 &y){ return x.getValue() > static_cast<int64_t>(y.getValue()); }
bool operator>=(const int64 &x, const int8 &y){ return x.getValue() >= static_cast<int64_t>(y.getValue()); }
bool operator<(const int64 &x, const int8 &y){ return x.getValue() < static_cast<int64_t>(y.getValue()); }
bool operator<=(const int64 &x, const int8 &y){ return x.getValue() <= static_cast<int64_t>(y.getValue()); }
int64 operator<<(const int64 &x, const int8 &y){ return int64(x.getValue() << static_cast<int64_t>(y.getValue())); }
int64 operator>>(const int64 &x, const int8 &y){ return int64(x.getValue() >> static_cast<int64_t>(y.getValue())); }
int64 operator&(const int64 &x, const int8 &y){ return int64(x.getValue() & static_cast<int64_t>(y.getValue())); }
int64 operator|(const int64 &x, const int8 &y){ return int64(x.getValue() | static_cast<int64_t>(y.getValue())); }
int64 operator^(const int64 &x, const int8 &y){ return int64(x.getValue() ^ static_cast<int64_t>(y.getValue())); }

// functions for int64 with int16
int64 operator+(const int64 &x, const int16 &y){ return int64(x.getValue() + static_cast<int64_t>(y.getValue())); }
int64 operator-(const int64 &x, const int16 &y){ return int64(x.getValue() - static_cast<int64_t>(y.getValue())); }
int64 operator*(const int64 &x, const int16 &y){ return int64(x.getValue() * static_cast<int64_t>(y.getValue())); }
float64 operator/(const int64 &x, const int16 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(static_cast<double>(x.getValue()) / static_cast<double>(y.getValue()));
}
int64 floordiv(const int64 &x, const int16 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(x.getValue() + static_cast<int64_t>(y.getValue()));
}
int64 operator%(const int64 &x, const int16 &y){ return int64(x.getValue() % static_cast<int64_t>(y.getValue())); }
int64 power(const int64 &x, const int16 &y){
  if(y.getValue() < 0) throw std::runtime_error("NegativeExponentError");
  int64_t res = 1;
  for(int16_t i = 0; i < y.getValue(); ++i){ res *= x.getValue(); }
  return int64(res);
}
bool operator==(const int64 &x, const int16 &y){ return x.getValue() == static_cast<int64_t>(y.getValue()); }
bool operator!=(const int64 &x, const int16 &y){ return x.getValue() != static_cast<int64_t>(y.getValue()); }
bool operator>(const int64 &x, const int16 &y){ return x.getValue() > static_cast<int64_t>(y.getValue()); }
bool operator>=(const int64 &x, const int16 &y){ return x.getValue() >= static_cast<int64_t>(y.getValue()); }
bool operator<(const int64 &x, const int16 &y){ return x.getValue() < static_cast<int64_t>(y.getValue()); }
bool operator<=(const int64 &x, const int16 &y){ return x.getValue() <= static_cast<int64_t>(y.getValue()); }
int64 operator<<(const int64 &x, const int16 &y){ return int64(x.getValue() << static_cast<int64_t>(y.getValue())); }
int64 operator>>(const int64 &x, const int16 &y){ return int64(x.getValue() >> static_cast<int64_t>(y.getValue())); }
int64 operator&(const int64 &x, const int16 &y){ return int64(x.getValue() & static_cast<int64_t>(y.getValue())); }
int64 operator|(const int64 &x, const int16 &y){ return int64(x.getValue() | static_cast<int64_t>(y.getValue())); }
int64 operator^(const int64 &x, const int16 &y){ return int64(x.getValue() ^ static_cast<int64_t>(y.getValue())); }

// functions for int64 with int32
int64 operator+(const int64 &x, const int32 &y){ return int64(x.getValue() + static_cast<int64_t>(y.getValue())); }
int64 operator-(const int64 &x, const int32 &y){ return int64(x.getValue() - static_cast<int64_t>(y.getValue())); }
int64 operator*(const int64 &x, const int32 &y){ return int64(x.getValue() * static_cast<int64_t>(y.getValue())); }
float64 operator/(const int64 &x, const int32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(static_cast<double>(x.getValue()) / static_cast<double>(y.getValue()));
}
int64 floordiv(const int64 &x, const int32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(x.getValue() + static_cast<int64_t>(y.getValue()));
}
int64 operator%(const int64 &x, const int32 &y){ return int64(x.getValue() % static_cast<int64_t>(y.getValue())); }
int64 power(const int64 &x, const int32 &y){
  if(y.getValue() < 0) throw std::runtime_error("NegativeExponentError");
  int64_t res = 1;
  for(int32_t i = 0; i < y.getValue(); ++i){ res *= x.getValue(); }
  return int64(res);
}
bool operator==(const int64 &x, const int32 &y){ return x.getValue() == static_cast<int64_t>(y.getValue()); }
bool operator!=(const int64 &x, const int32 &y){ return x.getValue() != static_cast<int64_t>(y.getValue()); }
bool operator>(const int64 &x, const int32 &y){ return x.getValue() > static_cast<int64_t>(y.getValue()); }
bool operator>=(const int64 &x, const int32 &y){ return x.getValue() >= static_cast<int64_t>(y.getValue()); }
bool operator<(const int64 &x, const int32 &y){ return x.getValue() < static_cast<int64_t>(y.getValue()); }
bool operator<=(const int64 &x, const int32 &y){ return x.getValue() <= static_cast<int64_t>(y.getValue()); }
int64 operator<<(const int64 &x, const int32 &y){ return int64(x.getValue() << static_cast<int64_t>(y.getValue())); }
int64 operator>>(const int64 &x, const int32 &y){ return int64(x.getValue() >> static_cast<int64_t>(y.getValue())); }
int64 operator&(const int64 &x, const int32 &y){ return int64(x.getValue() & static_cast<int64_t>(y.getValue())); }
int64 operator|(const int64 &x, const int32 &y){ return int64(x.getValue() | static_cast<int64_t>(y.getValue())); }
int64 operator^(const int64 &x, const int32 &y){ return int64(x.getValue() ^ static_cast<int64_t>(y.getValue())); }

// functions for int64 with float32
float64 operator+(const int64 &x, const float32 &y){ return float64(static_cast<double>(x.getValue()) + static_cast<double>(y.getValue())); }
float64 operator-(const int64 &x, const float32 &y){ return float64(static_cast<double>(x.getValue()) - static_cast<double>(y.getValue())); }
float64 operator*(const int64 &x, const float32 &y){ return float64(static_cast<double>(x.getValue()) * static_cast<double>(y.getValue())); }
float64 operator/(const int64 &x, const float32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(static_cast<double>(x.getValue()) / static_cast<double>(y.getValue()));
}
float64 floordiv(const int64 &x, const float32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(x.getValue() + static_cast<int64_t>(y.getValue()));
}
float64 operator%(const int64 &x, const float32 &y){ return float64(std::fmod(static_cast<double>(x.getValue()),static_cast<double>(y.getValue()))); }
float64 power(const int64 &x, const float32 &y){
  if(y.getValue() < 0) throw std::runtime_error("NegativeExponentError");
  return float64(std::pow(static_cast<double>(x.getValue()),static_cast<double>(y.getValue())));
}
bool operator==(const int64 &x, const float32 &y){ return static_cast<double>(x.getValue()) == static_cast<double>(y.getValue()); }
bool operator!=(const int64 &x, const float32 &y){ return static_cast<double>(x.getValue()) != static_cast<double>(y.getValue()); }
bool operator>(const int64 &x, const float32 &y){ return static_cast<double>(x.getValue()) > static_cast<double>(y.getValue()); }
bool operator>=(const int64 &x, const float32 &y){ return static_cast<double>(x.getValue()) >= static_cast<double>(y.getValue()); }
bool operator<(const int64 &x, const float32 &y){ return static_cast<double>(x.getValue()) < static_cast<double>(y.getValue()); }
bool operator<=(const int64 &x, const float32 &y){ return static_cast<double>(x.getValue()) <= static_cast<double>(y.getValue()); }

// functions for int64 with float64
float64 operator+(const int64 &x, const float64 &y){ return float64(static_cast<double>(x.getValue()) + static_cast<double>(y.getValue())); }
float64 operator-(const int64 &x, const float64 &y){ return float64(static_cast<double>(x.getValue()) - static_cast<double>(y.getValue())); }
float64 operator*(const int64 &x, const float64 &y){ return float64(static_cast<double>(x.getValue()) * static_cast<double>(y.getValue())); }
float64 operator/(const int64 &x, const float64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(static_cast<double>(x.getValue()) / static_cast<double>(y.getValue()));
}
float64 floordiv(const int64 &x, const float64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(x.getValue() + static_cast<int64_t>(y.getValue()));
}
float64 operator%(const int64 &x, const float64 &y){ return float64(std::fmod(static_cast<double>(x.getValue()),static_cast<double>(y.getValue()))); }
float64 power(const int64 &x, const float64 &y){
  if(y.getValue() < 0) throw std::runtime_error("NegativeExponentError");
  return float64(std::pow(static_cast<double>(x.getValue()),static_cast<double>(y.getValue())));
}
bool operator==(const int64 &x, const float64 &y){ return static_cast<double>(x.getValue()) == static_cast<double>(y.getValue()); }
bool operator!=(const int64 &x, const float64 &y){ return static_cast<double>(x.getValue()) != static_cast<double>(y.getValue()); }
bool operator>(const int64 &x, const float64 &y){ return static_cast<double>(x.getValue()) > static_cast<double>(y.getValue()); }
bool operator>=(const int64 &x, const float64 &y){ return static_cast<double>(x.getValue()) >= static_cast<double>(y.getValue()); }
bool operator<(const int64 &x, const float64 &y){ return static_cast<double>(x.getValue()) < static_cast<double>(y.getValue()); }
bool operator<=(const int64 &x, const float64 &y){ return static_cast<double>(x.getValue()) <= static_cast<double>(y.getValue()); }

// functions for int64 with complex128
complex128 operator+(const int64 &x, const complex128 &y){ return complex128(static_cast<double>(x.getValue()) + y.getValue().real(),y.getValue().imag()); }
complex128 operator-(const int64 &x, const complex128 &y){ return complex128(static_cast<double>(x.getValue()) - y.getValue().real(),-y.getValue().imag()); }
complex128 operator*(const int64 &x, const complex128 &y){ return complex128(static_cast<double>(x.getValue()) * y.getValue().real(),static_cast<double>(x.getValue()) * y.getValue().imag()); }
complex128 operator/(const int64 &x, const complex128 &y){
  if(y.getValue() == std::complex<double>(0,0)) throw std::runtime_error("ZeroDivisionError");
  return complex128(static_cast<double>(x.getValue()) / y.getValue().real(),static_cast<double>(x.getValue()) / y.getValue().imag());
}
complex128 power(const int64 &x, const complex128 &y){ return complex128(std::pow(static_cast<double>(x.getValue()),y.getValue())); }
bool operator==(const int64 &x, const complex128 &y){ return static_cast<double>(x.getValue()) == y.getValue().real(); }
bool operator!=(const int64 &x, const complex128 &y){ return static_cast<double>(x.getValue()) != y.getValue().real(); }

// functions for float32 builtin dtypes
float64 operator+(const float32 &x, const long &y){ return float64(static_cast<double>(x.getValue()) + static_cast<double>(y)); }
float64 operator-(const float32 &x, const long &y){ return float64(static_cast<double>(x.getValue()) - static_cast<double>(y)); }
float64 operator*(const float32 &x, const long &y){ return float64(static_cast<double>(x.getValue()) * static_cast<double>(y)); }
float64 operator/(const float32 &x, const long &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(static_cast<double>(x.getValue()) + static_cast<double>(y));
}
int64 floordiv(const float32 &x, const long &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(std::floor(static_cast<int64_t>(x.getValue()) / static_cast<int64_t>(y)));
}
float64 operator%(const float32 &x, const long &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(std::fmod(static_cast<double>(x.getValue()),static_cast<double>(y)));
}
float64 power(const float32 &x, const long &y){ return float64(std::pow(static_cast<double>(x.getValue()),static_cast<double>(y))); }
bool operator==(const float32 &x, const long &y){ return static_cast<double>(x.getValue()) == static_cast<double>(y); }
bool operator!=(const float32 &x, const long &y){ return static_cast<double>(x.getValue()) != static_cast<double>(y); }
bool operator>(const float32 &x, const long &y){ return static_cast<double>(x.getValue()) > static_cast<double>(y); }
bool operator>=(const float32 &x, const long &y){ return static_cast<double>(x.getValue()) >= static_cast<double>(y); }
bool operator<(const float32 &x, const long &y){ return static_cast<double>(x.getValue()) < static_cast<double>(y); }
bool operator<=(const float32 &x, const long &y){ return static_cast<double>(x.getValue()) <= static_cast<double>(y); }

float64 operator+(const float32 &x, const double &y){ return float64(static_cast<double>(x.getValue()) + static_cast<double>(y)); }
float64 operator-(const float32 &x, const double &y){ return float64(static_cast<double>(x.getValue()) - static_cast<double>(y)); }
float64 operator*(const float32 &x, const double &y){ return float64(static_cast<double>(x.getValue()) * static_cast<double>(y)); }
float64 operator/(const float32 &x, const double &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(static_cast<double>(x.getValue()) + static_cast<double>(y));
}
int64 floordiv(const float32 &x, const double &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(std::floor(static_cast<int64_t>(x.getValue()) / static_cast<int64_t>(y)));
}
float64 operator%(const float32 &x, const double &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(std::fmod(static_cast<double>(x.getValue()),static_cast<double>(y)));
}
float64 power(const float32 &x, const double &y){ return float64(std::pow(static_cast<double>(x.getValue()),static_cast<double>(y))); }
bool operator==(const float32 &x, const double &y){ return static_cast<double>(x.getValue()) == y; }
bool operator!=(const float32 &x, const double &y){ return static_cast<double>(x.getValue()) != y; }
bool operator>(const float32 &x, const double &y){ return static_cast<double>(x.getValue()) > y; }
bool operator>=(const float32 &x, const double &y){ return static_cast<double>(x.getValue()) >= y; }
bool operator<(const float32 &x, const double &y){ return static_cast<double>(x.getValue()) < y; }
bool operator<=(const float32 &x, const double &y){ return static_cast<double>(x.getValue()) <= y; }

complex128 operator+(const float32 &x, const std::complex<double> &y){ return complex128(static_cast<double>(x.getValue()) + y.real(),y.imag()); }
complex128 operator-(const float32 &x, const std::complex<double> &y){ return complex128(static_cast<double>(x.getValue()) - y.real(),-y.imag()); }
complex128 operator*(const float32 &x, const std::complex<double> &y){ return complex128(static_cast<double>(x.getValue()) * y.real(),static_cast<double>(x.getValue()) * y.imag()); }
complex128 operator/(const float32 &x, const std::complex<double> &y){
  if(y == std::complex<double>(0,0)) throw std::runtime_error("ZeroDivisionError");
  return complex128(static_cast<double>(x.getValue()) / y.real(),static_cast<double>(x.getValue()) / y.imag());
}
complex128 power(const float32 &x, const std::complex<double> &y){ return complex128(std::pow(static_cast<double>(x.getValue()),y)); }
bool operator==(const float32 &x, const std::complex<double> &y){ return static_cast<double>(x.getValue()) == y.real(); }
bool operator!=(const float32 &x, const std::complex<double> &y){ return static_cast<double>(x.getValue()) != y.real(); }

// functions for float32 with int8
float32 operator+(const float32 &x, const int8 &y){ return float32(x.getValue() + static_cast<float>(y.getValue())); }
float32 operator-(const float32 &x, const int8 &y){ return float32(x.getValue() - static_cast<float>(y.getValue())); }
float32 operator*(const float32 &x, const int8 &y){ return float32(x.getValue() * static_cast<float>(y.getValue())); }
float32 operator/(const float32 &x, const int8 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float32(x.getValue() / static_cast<float>(y.getValue()));
}
int32 floordiv(const float32 &x, const int8 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int32(static_cast<int32_t>(x.getValue()) / static_cast<int32_t>(y.getValue()));
}
float32 operator%(const float32 &x, const int8 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float32(std::fmod(x.getValue(),static_cast<float>(y.getValue())));
}
float32 power(const float32 &x, const int8 &y){ return float32(std::pow(x.getValue(),static_cast<float>(y.getValue()))); }
bool operator==(const float32 &x, const int8 &y){ return x.getValue() == static_cast<float>(y.getValue()); }
bool operator!=(const float32 &x, const int8 &y){ return x.getValue() != static_cast<float>(y.getValue()); }
bool operator>(const float32 &x, const int8 &y){ return x.getValue() > static_cast<float>(y.getValue()); }
bool operator>=(const float32 &x, const int8 &y){ return x.getValue() >= static_cast<float>(y.getValue()); }
bool operator<(const float32 &x, const int8 &y){ return x.getValue() < static_cast<float>(y.getValue()); }
bool operator<=(const float32 &x, const int8 &y){ return x.getValue() <= static_cast<float>(y.getValue()); }

// functions for float32 with int16
float32 operator+(const float32 &x, const int16 &y){ return float32(x.getValue() + static_cast<float>(y.getValue())); }
float32 operator-(const float32 &x, const int16 &y){ return float32(x.getValue() - static_cast<float>(y.getValue())); }
float32 operator*(const float32 &x, const int16 &y){ return float32(x.getValue() * static_cast<float>(y.getValue())); }
float32 operator/(const float32 &x, const int16 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float32(x.getValue() / static_cast<float>(y.getValue()));
}
int32 floordiv(const float32 &x, const int16 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int32(static_cast<int32_t>(x.getValue()) / static_cast<int32_t>(y.getValue()));
}
float32 operator%(const float32 &x, const int16 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float32(std::fmod(x.getValue(),static_cast<float>(y.getValue())));
}
float32 power(const float32 &x, const int16 &y){ return float32(std::pow(x.getValue(),static_cast<float>(y.getValue()))); }
bool operator==(const float32 &x, const int16 &y){ return x.getValue() == static_cast<float>(y.getValue()); }
bool operator!=(const float32 &x, const int16 &y){ return x.getValue() != static_cast<float>(y.getValue()); }
bool operator>(const float32 &x, const int16 &y){ return x.getValue() > static_cast<float>(y.getValue()); }
bool operator>=(const float32 &x, const int16 &y){ return x.getValue() >= static_cast<float>(y.getValue()); }
bool operator<(const float32 &x, const int16 &y){ return x.getValue() < static_cast<float>(y.getValue()); }
bool operator<=(const float32 &x, const int16 &y){ return x.getValue() <= static_cast<float>(y.getValue()); }

// functions for float32 with int32
float32 operator+(const float32 &x, const int32 &y){ return float32(x.getValue() + static_cast<float>(y.getValue())); }
float32 operator-(const float32 &x, const int32 &y){ return float32(x.getValue() - static_cast<float>(y.getValue())); }
float32 operator*(const float32 &x, const int32 &y){ return float32(x.getValue() * static_cast<float>(y.getValue())); }
float32 operator/(const float32 &x, const int32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float32(x.getValue() / static_cast<float>(y.getValue()));
}
int32 floordiv(const float32 &x, const int32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int32(static_cast<int32_t>(x.getValue()) / static_cast<int32_t>(y.getValue()));
}
float32 operator%(const float32 &x, const int32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float32(std::fmod(x.getValue(),static_cast<float>(y.getValue())));
}
float32 power(const float32 &x, const int32 &y){ return float32(std::pow(x.getValue(),static_cast<float>(y.getValue()))); }
bool operator==(const float32 &x, const int32 &y){ return x.getValue() == static_cast<float>(y.getValue()); }
bool operator!=(const float32 &x, const int32 &y){ return x.getValue() != static_cast<float>(y.getValue()); }
bool operator>(const float32 &x, const int32 &y){ return x.getValue() > static_cast<float>(y.getValue()); }
bool operator>=(const float32 &x, const int32 &y){ return x.getValue() >= static_cast<float>(y.getValue()); }
bool operator<(const float32 &x, const int32 &y){ return x.getValue() < static_cast<float>(y.getValue()); }
bool operator<=(const float32 &x, const int32 &y){ return x.getValue() <= static_cast<float>(y.getValue()); }

// functions for float32 with int64
float64 operator+(const float32 &x, const int64 &y){ return float64(static_cast<double>(x.getValue()) + static_cast<double>(y.getValue())); }
float64 operator-(const float32 &x, const int64 &y){ return float64(static_cast<double>(x.getValue()) - static_cast<double>(y.getValue())); }
float64 operator*(const float32 &x, const int64 &y){ return float64(static_cast<double>(x.getValue()) * static_cast<double>(y.getValue())); }
float64 operator/(const float32 &x, const int64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(static_cast<double>(x.getValue()) / static_cast<double>(y.getValue()));
}
int64 floordiv(const float32 &x, const int64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(static_cast<int64_t>(x.getValue()) / static_cast<int64_t>(y.getValue()));
}
float64 operator%(const float32 &x, const int64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(std::fmod(static_cast<double>(x.getValue()),static_cast<double>(y.getValue())));
}
float64 power(const float32 &x, const int64 &y){ return float64(std::pow(static_cast<double>(x.getValue()),static_cast<double>(y.getValue()))); }
bool operator==(const float32 &x, const int64 &y){ return static_cast<double>(x.getValue()) == static_cast<double>(y.getValue()); }
bool operator!=(const float32 &x, const int64 &y){ return static_cast<double>(x.getValue()) != static_cast<double>(y.getValue()); }
bool operator>(const float32 &x, const int64 &y){ return static_cast<double>(x.getValue()) > static_cast<double>(y.getValue()); }
bool operator>=(const float32 &x, const int64 &y){ return static_cast<double>(x.getValue()) >= static_cast<double>(y.getValue()); }
bool operator<(const float32 &x, const int64 &y){ return static_cast<double>(x.getValue()) < static_cast<double>(y.getValue()); }
bool operator<=(const float32 &x, const int64 &y){ return static_cast<double>(x.getValue()) <= static_cast<double>(y.getValue()); }

// functions for float32 with float64
float64 operator+(const float32 &x, const float64 &y){ return float64(static_cast<double>(x.getValue()) + y.getValue()); }
float64 operator-(const float32 &x, const float64 &y){ return float64(static_cast<double>(x.getValue()) - y.getValue()); }
float64 operator*(const float32 &x, const float64 &y){ return float64(static_cast<double>(x.getValue()) * y.getValue()); }
float64 operator/(const float32 &x, const float64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(static_cast<double>(x.getValue()) / y.getValue());
}
int64 floordiv(const float32 &x, const float64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(static_cast<int64_t>(x.getValue()) / static_cast<int64_t>(y.getValue()));
}
float64 operator%(const float32 &x, const float64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(std::fmod(static_cast<double>(x.getValue()),y.getValue()));
}
float64 power(const float32 &x, const float64 &y){ return float64(std::pow(static_cast<double>(x.getValue()),y.getValue())); }
bool operator==(const float32 &x, const float64 &y){ return static_cast<double>(x.getValue()) == y.getValue(); }
bool operator!=(const float32 &x, const float64 &y){ return static_cast<double>(x.getValue()) != y.getValue(); }
bool operator>(const float32 &x, const float64 &y){ return static_cast<double>(x.getValue()) > static_cast<double>(y.getValue()); }
bool operator>=(const float32 &x, const float64 &y){ return static_cast<double>(x.getValue()) >= y.getValue(); }
bool operator<(const float32 &x, const float64 &y){ return static_cast<double>(x.getValue()) < y.getValue(); }
bool operator<=(const float32 &x, const float64 &y){ return static_cast<double>(x.getValue()) <= y.getValue(); }

// functions for float32 with complex128
complex128 operator+(const float32 &x, const complex128 &y){ return complex128(static_cast<double>(x.getValue()) + y.getValue().real(),y.getValue().imag()); }
complex128 operator-(const float32 &x, const complex128 &y){ return complex128(static_cast<double>(x.getValue()) - y.getValue().real(),-y.getValue().imag()); }
complex128 operator*(const float32 &x, const complex128 &y){ return complex128(static_cast<double>(x.getValue()) * y.getValue().real(),static_cast<double>(x.getValue()) * y.getValue().imag()); }
complex128 operator/(const float32 &x, const complex128 &y){
  if(y.getValue() == std::complex<double>(0,0)) throw std::runtime_error("ZeroDivisionError");
  return complex128(static_cast<double>(x.getValue()) / y.getValue().real(),static_cast<double>(x.getValue()) / y.getValue().imag());
}
complex128 power(const float32 &x, const complex128 &y){ return complex128(std::pow(static_cast<double>(x.getValue()),y.getValue())); }
bool operator==(const float32 &x, const complex128 &y){ return static_cast<double>(x.getValue()) == y.getValue().real(); }
bool operator!=(const float32 &x, const complex128 &y){ return static_cast<double>(x.getValue()) != y.getValue().real(); }

// functions for float64 with builtin dtypes
float64 operator+(const float64 &x, const long &y){ return float64(x.getValue() + static_cast<double>(y)); }
float64 operator-(const float64 &x, const long &y){ return float64(x.getValue() - static_cast<double>(y)); }
float64 operator*(const float64 &x, const long &y){ return float64(x.getValue() * static_cast<double>(y)); }
float64 operator/(const float64 &x, const long &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(x.getValue() / static_cast<double>(y));
}
int64 floordiv(const float64 &x, const long &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(static_cast<long>(x.getValue()) / static_cast<long>(y));
}
float64 operator%(const float64 &x, const long &y){ return float64(std::fmod(x.getValue(),static_cast<double>(y))); }
float64 power(const float64 &x, const long &y){ return float64(std::pow(x.getValue(),static_cast<double>(y))); }
bool operator==(const float64 &x, const long &y){ return x.getValue() == static_cast<double>(y); }
bool operator!=(const float64 &x, const long &y){ return x.getValue() != static_cast<double>(y); }
bool operator>(const float64 &x, const long &y){ return x.getValue() > static_cast<double>(y); }
bool operator>=(const float64 &x, const long &y){ return x.getValue() >= static_cast<double>(y); }
bool operator<(const float64 &x, const long &y){ return x.getValue() < static_cast<double>(y); }
bool operator<=(const float64 &x, const long &y){ return x.getValue() <= static_cast<double>(y); }

float64 operator+(const float64 &x, const double &y){ return float64(x.getValue() + y); }
float64 operator-(const float64 &x, const double &y){ return float64(x.getValue() - y); }
float64 operator*(const float64 &x, const double &y){ return float64(x.getValue() * y); }
float64 operator/(const float64 &x, const double &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(x.getValue() / y);
}
int64 floordiv(const float64 &x, const double &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(static_cast<long>(x.getValue()) / static_cast<long>(y));
}
float64 operator%(const float64 &x, const double &y){ return float64(std::fmod(x.getValue(),y)); }
float64 power(const float64 &x, const double &y){ return float64(std::pow(x.getValue(),y)); }
bool operator==(const float64 &x, const double &y){ return x.getValue() == y; }
bool operator!=(const float64 &x, const double &y){ return x.getValue() != y; }
bool operator>(const float64 &x, const double &y){ return x.getValue() > y; }
bool operator>=(const float64 &x, const double &y){ return x.getValue() >= y; }
bool operator<(const float64 &x, const double &y){ return x.getValue() < y; }
bool operator<=(const float64 &x, const double &y){ return x.getValue() <= y; }

complex128 operator+(const float64 &x, const std::complex<double> &y){ return complex128(x.getValue() + y.real(),y.imag()); }
complex128 operator-(const float64 &x, const std::complex<double> &y){ return complex128(x.getValue() - y.real(),-y.imag()); }
complex128 operator*(const float64 &x, const std::complex<double> &y){ return complex128(x.getValue() * y.real(),x.getValue() * y.imag()); }
complex128 operator/(const float64 &x, const std::complex<double> &y){
  if(y == std::complex<double>(0,0)) throw std::runtime_error("ZeroDivisionError");
  return complex128(x.getValue() / y.real(),x.getValue() / y.imag());
}
complex128 power(const float64 &x, const std::complex<double> &y){ return complex128(std::pow(x.getValue(),y)); }
bool operator==(const float64 &x, const std::complex<double> &y){ return x.getValue() == y.real(); }
bool operator!=(const float64 &x, const std::complex<double> &y){ return x.getValue() != y.real(); }

// functions for float64 with int8
float64 operator+(const float64 &x, const int8 &y){ return float64(x.getValue() + static_cast<double>(y.getValue())); }
float64 operator-(const float64 &x, const int8 &y){ return float64(x.getValue() - static_cast<double>(y.getValue())); }
float64 operator*(const float64 &x, const int8 &y){ return float64(x.getValue() * static_cast<double>(y.getValue())); }
float64 operator/(const float64 &x, const int8 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(x.getValue() / static_cast<double>(y.getValue()));
}
int64 floordiv(const float64 &x, const int8 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(static_cast<long>(x.getValue()) + static_cast<long>(y.getValue()));
}
float64 operator%(const float64 &x, const int8 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(std::fmod(x.getValue(),static_cast<double>(y.getValue())));
}
float64 power(const float64 &x, const int8 &y){ return float64(std::pow(x.getValue(),static_cast<double>(y.getValue()))); }
bool operator==(const float64 &x, const int8 &y){ return x.getValue() == static_cast<double>(y.getValue()); }
bool operator!=(const float64 &x, const int8 &y){ return x.getValue() != static_cast<double>(y.getValue()); }
bool operator>(const float64 &x, const int8 &y){ return x.getValue() > static_cast<double>(y.getValue()); }
bool operator>=(const float64 &x, const int8 &y){ return x.getValue() >= static_cast<double>(y.getValue()); }
bool operator<(const float64 &x, const int8 &y){ return x.getValue() < static_cast<double>(y.getValue()); }
bool operator<=(const float64 &x, const int8 &y){ return x.getValue() <= static_cast<double>(y.getValue()); }

// functions for float64 with int16
float64 operator+(const float64 &x, const int16 &y){ return float64(x.getValue() + static_cast<double>(y.getValue())); }
float64 operator-(const float64 &x, const int16 &y){ return float64(x.getValue() - static_cast<double>(y.getValue())); }
float64 operator*(const float64 &x, const int16 &y){ return float64(x.getValue() * static_cast<double>(y.getValue())); }
float64 operator/(const float64 &x, const int16 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(x.getValue() / static_cast<double>(y.getValue()));
}
int64 floordiv(const float64 &x, const int16 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(static_cast<long>(x.getValue()) + static_cast<long>(y.getValue()));
}
float64 operator%(const float64 &x, const int16 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(std::fmod(x.getValue(),static_cast<double>(y.getValue())));
}
float64 power(const float64 &x, const int16 &y){ return float64(std::pow(x.getValue(),static_cast<double>(y.getValue()))); }
bool operator==(const float64 &x, const int16 &y){ return x.getValue() == static_cast<double>(y.getValue()); }
bool operator!=(const float64 &x, const int16 &y){ return x.getValue() != static_cast<double>(y.getValue()); }
bool operator>(const float64 &x, const int16 &y){ return x.getValue() > static_cast<double>(y.getValue()); }
bool operator>=(const float64 &x, const int16 &y){ return x.getValue() >= static_cast<double>(y.getValue()); }
bool operator<(const float64 &x, const int16 &y){ return x.getValue() < static_cast<double>(y.getValue()); }
bool operator<=(const float64 &x, const int16 &y){ return x.getValue() <= static_cast<double>(y.getValue()); }

// functions for float64 with int32
float64 operator+(const float64 &x, const int32 &y){ return float64(x.getValue() + static_cast<double>(y.getValue())); }
float64 operator-(const float64 &x, const int32 &y){ return float64(x.getValue() - static_cast<double>(y.getValue())); }
float64 operator*(const float64 &x, const int32 &y){ return float64(x.getValue() * static_cast<double>(y.getValue())); }
float64 operator/(const float64 &x, const int32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(x.getValue() / static_cast<double>(y.getValue()));
}
int64 floordiv(const float64 &x, const int32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(static_cast<long>(x.getValue()) + static_cast<long>(y.getValue()));
}
float64 operator%(const float64 &x, const int32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(std::fmod(x.getValue(),static_cast<double>(y.getValue())));
}
float64 power(const float64 &x, const int32 &y){ return float64(std::pow(x.getValue(),static_cast<double>(y.getValue()))); }
bool operator==(const float64 &x, const int32 &y){ return x.getValue() == static_cast<double>(y.getValue()); }
bool operator!=(const float64 &x, const int32 &y){ return x.getValue() != static_cast<double>(y.getValue()); }
bool operator>(const float64 &x, const int32 &y){ return x.getValue() > static_cast<double>(y.getValue()); }
bool operator>=(const float64 &x, const int32 &y){ return x.getValue() >= static_cast<double>(y.getValue()); }
bool operator<(const float64 &x, const int32 &y){ return x.getValue() < static_cast<double>(y.getValue()); }
bool operator<=(const float64 &x, const int32 &y){ return x.getValue() <= static_cast<double>(y.getValue()); }

// functions for float64 with int64
float64 operator+(const float64 &x, const int64 &y){ return float64(x.getValue() + static_cast<double>(y.getValue())); }
float64 operator-(const float64 &x, const int64 &y){ return float64(x.getValue() - static_cast<double>(y.getValue())); }
float64 operator*(const float64 &x, const int64 &y){ return float64(x.getValue() * static_cast<double>(y.getValue())); }
float64 operator/(const float64 &x, const int64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(x.getValue() / static_cast<double>(y.getValue()));
}
int64 floordiv(const float64 &x, const int64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(static_cast<long>(x.getValue()) + static_cast<long>(y.getValue()));
}
float64 operator%(const float64 &x, const int64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(std::fmod(x.getValue(),static_cast<double>(y.getValue())));
}
float64 power(const float64 &x, const int64 &y){ return float64(std::pow(x.getValue(),static_cast<double>(y.getValue()))); }
bool operator==(const float64 &x, const int64 &y){ return x.getValue() == static_cast<double>(y.getValue()); }
bool operator!=(const float64 &x, const int64 &y){ return x.getValue() != static_cast<double>(y.getValue()); }
bool operator>(const float64 &x, const int64 &y){ return x.getValue() > static_cast<double>(y.getValue()); }
bool operator>=(const float64 &x, const int64 &y){ return x.getValue() >= static_cast<double>(y.getValue()); }
bool operator<(const float64 &x, const int64 &y){ return x.getValue() < static_cast<double>(y.getValue()); }
bool operator<=(const float64 &x, const int64 &y){ return x.getValue() <= static_cast<double>(y.getValue()); }

// functions for float64 with float32
float64 operator+(const float64 &x, const float32 &y){ return float64(x.getValue() + static_cast<double>(y.getValue())); }
float64 operator-(const float64 &x, const float32 &y){ return float64(x.getValue() - static_cast<double>(y.getValue())); }
float64 operator*(const float64 &x, const float32 &y){ return float64(x.getValue() * static_cast<double>(y.getValue())); }
float64 operator/(const float64 &x, const float32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(x.getValue() / static_cast<double>(y.getValue()));
}
int64 floordiv(const float64 &x, const float32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return int64(static_cast<long>(x.getValue()) + static_cast<long>(y.getValue()));
}
float64 operator%(const float64 &x, const float32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return float64(std::fmod(x.getValue(),static_cast<double>(y.getValue())));
}
float64 power(const float64 &x, const float32 &y){ return float64(std::pow(x.getValue(),static_cast<double>(y.getValue()))); }
bool operator==(const float64 &x, const float32 &y){ return x.getValue() == static_cast<double>(y.getValue()); }
bool operator!=(const float64 &x, const float32 &y){ return x.getValue() != static_cast<double>(y.getValue()); }
bool operator>(const float64 &x, const float32 &y){ return x.getValue() > static_cast<double>(y.getValue()); }
bool operator>=(const float64 &x, const float32 &y){ return x.getValue() >= static_cast<double>(y.getValue()); }
bool operator<(const float64 &x, const float32 &y){ return x.getValue() < static_cast<double>(y.getValue()); }
bool operator<=(const float64 &x, const float32 &y){ return x.getValue() <= static_cast<double>(y.getValue()); }

// functions for float64 with complex128
complex128 operator+(const float64 &x, const complex128 &y){ return complex128(x.getValue() + y.getValue().real(),y.getValue().imag()); }
complex128 operator-(const float64 &x, const complex128 &y){ return complex128(x.getValue() - y.getValue().real(),-y.getValue().imag()); }
complex128 operator*(const float64 &x, const complex128 &y){ return complex128(x.getValue() * y.getValue().real(),x.getValue() * y.getValue().imag()); }
complex128 operator/(const float64 &x, const complex128 &y){
  if(y.getValue() == std::complex<double>(0,0)) throw std::runtime_error("ZeroDivisionError");
  return complex128(x.getValue() / y.getValue().real(),x.getValue() / y.getValue().imag());
}
complex128 power(const float64 &x, const complex128 &y){ return complex128(std::pow(x.getValue(),y.getValue())); }
bool operator==(const float64 &x, const complex128 &y){ return x.getValue() == y.getValue().real(); }
bool operator!=(const float64 &x, const complex128 &y){ return x.getValue() != y.getValue().real(); }

// functions for complex128 with builtin dtypes
complex128 operator+(const complex128 &x, const long &y){ return complex128(x.real() + static_cast<double>(y), x.imag()); }
complex128 operator-(const complex128 &x, const long &y){ return complex128(x.real() - static_cast<double>(y), x.imag()); }
complex128 operator*(const complex128 &x, const long &y){ return complex128(x.real() * static_cast<double>(y), x.imag() * static_cast<double>(y)); }
complex128 operator/(const complex128 &x, const long &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return complex128(x.real() / static_cast<double>(y), x.imag() / static_cast<double>(y));
}
complex128 power(const complex128 &x, const long &y){ return complex128(std::pow(x.getValue(),static_cast<double>(y))); }
bool operator==(const complex128 &x, const long &y){ return x.getValue() == static_cast<double>(y); }
bool operator!=(const complex128 &x, const long &y){ return x.getValue() != static_cast<double>(y); }

complex128 operator+(const complex128 &x, const double &y){ return complex128(x.real() + y, x.imag()); }
complex128 operator-(const complex128 &x, const double &y){ return complex128(x.real() - y, x.imag()); }
complex128 operator*(const complex128 &x, const double &y){ return complex128(x.real() * y, x.imag() * y); }
complex128 operator/(const complex128 &x, const double &y){
  if(y == 0) throw std::runtime_error("ZeroDivisionError");
  return complex128(x.real() / y, x.imag() / y);
}
complex128 power(const complex128 &x, const double &y){ return complex128(std::pow(x.getValue(),y)); }
bool operator==(const complex128 &x, const double &y){ return x.getValue() == y; }
bool operator!=(const complex128 &x, const double &y){ return x.getValue() != y; }

complex128 operator+(const complex128 &x, const std::complex<double> &y){ return complex128(x.real() + y.real(), x.imag() + y.imag()); }
complex128 operator-(const complex128 &x, const std::complex<double> &y){ return complex128(x.real() - y.real(), x.imag() - y.imag()); }
complex128 operator*(const complex128 &x, const std::complex<double> &y){ return complex128(x.real() * y.real() - x.imag() * y.imag(), x.real() * y.imag() + x.imag() * y.real()); }
complex128 operator/(const complex128 &x, const std::complex<double> &y){
  if(y == std::complex<double>(0,0)) throw std::runtime_error("ZeroDivisionError");
  return complex128((x.real() * y.real() + x.imag() * y.imag()) / (y.real() * y.real() + y.imag() * y.imag()), (x.imag() * y.real() - x.real() * y.imag()) / (y.real() * y.real() + y.imag() * y.imag()));
}
complex128 power(const complex128 &x, const std::complex<double> &y){ return complex128(std::pow(x.getValue(),y)); }
bool operator==(const complex128 &x, const std::complex<double> &y){ return x.getValue() == y; }
bool operator!=(const complex128 &x, const std::complex<double> &y){ return x.getValue() != y; }

// functions for complex128 with int8
complex128 operator+(const complex128 &x, const int8 &y){ return complex128(x.real() + static_cast<double>(y.getValue()), x.imag()); }
complex128 operator-(const complex128 &x, const int8 &y){ return complex128(x.real() - static_cast<double>(y.getValue()), x.imag()); }
complex128 operator*(const complex128 &x, const int8 &y){ return complex128(x.real() * static_cast<double>(y.getValue()), x.imag() * static_cast<double>(y.getValue())); }
complex128 operator/(const complex128 &x, const int8 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return complex128(x.real() / static_cast<double>(y.getValue()), x.imag() / static_cast<double>(y.getValue()));
}
complex128 power(const complex128 &x, const int8 &y){ return complex128(std::pow(x.getValue(),static_cast<double>(y.getValue()))); }
bool operator==(const complex128 &x, const int8 &y){ return x.getValue() == static_cast<double>(y.getValue()); }
bool operator!=(const complex128 &x, const int8 &y){ return x.getValue() != static_cast<double>(y.getValue()); }

// functions for complex128 with int16
complex128 operator+(const complex128 &x, const int16 &y){ return complex128(x.real() + static_cast<double>(y.getValue()), x.imag()); }
complex128 operator-(const complex128 &x, const int16 &y){ return complex128(x.real() - static_cast<double>(y.getValue()), x.imag()); }
complex128 operator*(const complex128 &x, const int16 &y){ return complex128(x.real() * static_cast<double>(y.getValue()), x.imag() * static_cast<double>(y.getValue())); }
complex128 operator/(const complex128 &x, const int16 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return complex128(x.real() / static_cast<double>(y.getValue()), x.imag() / static_cast<double>(y.getValue()));
}
complex128 power(const complex128 &x, const int16 &y){ return complex128(std::pow(x.getValue(),static_cast<double>(y.getValue()))); }
bool operator==(const complex128 &x, const int16 &y){ return x.getValue() == static_cast<double>(y.getValue()); }
bool operator!=(const complex128 &x, const int16 &y){ return x.getValue() != static_cast<double>(y.getValue()); }

// functions for complex128 with int32
complex128 operator+(const complex128 &x, const int32 &y){ return complex128(x.real() + static_cast<double>(y.getValue()), x.imag()); }
complex128 operator-(const complex128 &x, const int32 &y){ return complex128(x.real() - static_cast<double>(y.getValue()), x.imag()); }
complex128 operator*(const complex128 &x, const int32 &y){ return complex128(x.real() * static_cast<double>(y.getValue()), x.imag() * static_cast<double>(y.getValue())); }
complex128 operator/(const complex128 &x, const int32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return complex128(x.real() / static_cast<double>(y.getValue()), x.imag() / static_cast<double>(y.getValue()));
}
complex128 power(const complex128 &x, const int32 &y){ return complex128(std::pow(x.getValue(),static_cast<double>(y.getValue()))); }
bool operator==(const complex128 &x, const int32 &y){ return x.getValue() == static_cast<double>(y.getValue()); }
bool operator!=(const complex128 &x, const int32 &y){ return x.getValue() != static_cast<double>(y.getValue()); }

// functions for complex128 with int64
complex128 operator+(const complex128 &x, const int64 &y){ return complex128(x.real() + static_cast<double>(y.getValue()), x.imag()); }
complex128 operator-(const complex128 &x, const int64 &y){ return complex128(x.real() - static_cast<double>(y.getValue()), x.imag()); }
complex128 operator*(const complex128 &x, const int64 &y){ return complex128(x.real() * static_cast<double>(y.getValue()), x.imag() * static_cast<double>(y.getValue())); }
complex128 operator/(const complex128 &x, const int64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return complex128(x.real() / static_cast<double>(y.getValue()), x.imag() / static_cast<double>(y.getValue()));
}
complex128 power(const complex128 &x, const int64 &y){ return complex128(std::pow(x.getValue(),static_cast<double>(y.getValue()))); }
bool operator==(const complex128 &x, const int64 &y){ return x.getValue() == static_cast<double>(y.getValue()); }
bool operator!=(const complex128 &x, const int64 &y){ return x.getValue() != static_cast<double>(y.getValue()); }

// functions for complex128 with float32
complex128 operator+(const complex128 &x, const float32 &y){ return complex128(x.real() + static_cast<double>(y.getValue()), x.imag()); }
complex128 operator-(const complex128 &x, const float32 &y){ return complex128(x.real() - static_cast<double>(y.getValue()), x.imag()); }
complex128 operator*(const complex128 &x, const float32 &y){ return complex128(x.real() * static_cast<double>(y.getValue()), x.imag() * static_cast<double>(y.getValue())); }
complex128 operator/(const complex128 &x, const float32 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return complex128(x.real() / static_cast<double>(y.getValue()), x.imag() / static_cast<double>(y.getValue()));
}
complex128 power(const complex128 &x, const float32 &y){ return complex128(std::pow(x.getValue(),static_cast<double>(y.getValue()))); }
bool operator==(const complex128 &x, const float32 &y){ return x.getValue() == static_cast<double>(y.getValue()); }
bool operator!=(const complex128 &x, const float32 &y){ return x.getValue() != static_cast<double>(y.getValue()); }

// functions for complex128 with float64
complex128 operator+(const complex128 &x, const float64 &y){ return complex128(x.real() + y.getValue(), x.imag()); }
complex128 operator-(const complex128 &x, const float64 &y){ return complex128(x.real() - y.getValue(), x.imag()); }
complex128 operator*(const complex128 &x, const float64 &y){ return complex128(x.real() * y.getValue(), x.imag() * y.getValue()); }
complex128 operator/(const complex128 &x, const float64 &y){
  if(y.getValue() == 0) throw std::runtime_error("ZeroDivisionError");
  return complex128(x.real() / y.getValue(), x.imag() / y.getValue());
}
complex128 power(const complex128 &x, const float64 &y){ return complex128(std::pow(x.getValue(),y.getValue())); }
bool operator==(const complex128 &x, const float64 &y){ return x.getValue() == y.getValue(); }
bool operator!=(const complex128 &x, const float64 &y){ return x.getValue() != y.getValue(); }




PYBIND11_MODULE(dtypes, m){
  pybind11::class_<int8>(m, "int8")
    .def(pybind11::init<int8_t>())
    .def("__repr__", &int8::repr)
    .def_property_readonly("nbits", &int8::nbits)
    .def("getValue", &int8::getValue)
    .def("__add__", &int8::operator+)
    .def("__add__", pybind11::overload_cast<const int8&, const long&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int8&, const double&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int8&, const std::complex<double>&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int8&, const int16&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int8&, const int32&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int8&, const int64&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int8&, const float32&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int8&, const float64&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int8&, const complex128&>(&operator+))
    .def("__sub__", &int8::operator-)
    .def("__sub__", pybind11::overload_cast<const int8&, const long&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const int8&, const double&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const int8&, const std::complex<double>&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const int8&, const int16&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const int8&, const int32&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const int8&, const int64&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const int8&, const float32&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const int8&, const float64&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const int8&, const complex128&>(&operator-))
    .def("__mul__", &int8::operator*)
    .def("__mul__", pybind11::overload_cast<const int8&, const long&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int8&, const double&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int8&, const std::complex<double>&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int8&, const int16&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int8&, const int32&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int8&, const int64&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int8&, const float32&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int8&, const float64&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int8&, const complex128&>(&operator*))
    .def("__truediv__", pybind11::overload_cast<const int8&, const int8&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int8&, const long&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int8&, const double&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int8&, const std::complex<double>&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int8&, const int16&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int8&, const int32&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int8&, const int64&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int8&, const float32&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int8&, const float64&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int8&, const complex128&>(&operator/))
    .def("__mod__", &int8::operator%)
    .def("__mod__", pybind11::overload_cast<const int8&, const long&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const int8&, const double&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const int8&, const int16&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const int8&, const int32&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const int8&, const int64&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const int8&, const float32&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const int8&, const float64&>(&operator%))
    .def("__floordiv__", &int8::floordiv)
    .def("__floordiv__", pybind11::overload_cast<const int8&, const long&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const int8&, const double&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const int8&, const int16&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const int8&, const int32&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const int8&, const int64&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const int8&, const float32&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const int8&, const float64&>(&floordiv))
    .def("__pow__", &int8::power)
    .def("__pow__", pybind11::overload_cast<const int8&, const long&>(&power))
    .def("__pow__", pybind11::overload_cast<const int8&, const double&>(&power))
    .def("__pow__", pybind11::overload_cast<const int8&, const std::complex<double>&>(&power))
    .def("__pow__", pybind11::overload_cast<const int8&, const int16&>(&power))
    .def("__pow__", pybind11::overload_cast<const int8&, const int32&>(&power))
    .def("__pow__", pybind11::overload_cast<const int8&, const int64&>(&power))
    .def("__pow__", pybind11::overload_cast<const int8&, const float32&>(&power))
    .def("__pow__", pybind11::overload_cast<const int8&, const float64&>(&power))
    .def("__pow__", pybind11::overload_cast<const int8&, const complex128&>(&power))
    .def("__abs__", &int8::abs)
    .def("__neg__", &int8::neg)
    .def("__pos__", &int8::pos)
    .def("__eq__", &int8::operator==)
    .def("__eq__", pybind11::overload_cast<const int8&, const long&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int8&, const double&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int8&, const std::complex<double>&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int8&, const int16&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int8&, const int32&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int8&, const int64&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int8&, const float32&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int8&, const float64&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int8&, const complex128&>(&operator==))
    .def("__ne__", &int8::operator!=)
    .def("__ne__", pybind11::overload_cast<const int8&, const long&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int8&, const double&>(&operator!=))
    .def("__neq__", pybind11::overload_cast<const int8&, const std::complex<double>&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int8&, const int16&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int8&, const int32&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int8&, const int64&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int8&, const float32&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int8&, const float64&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int8&, const complex128&>(&operator!=))
    .def("__lt__", &int8::operator<)
    .def("__lt__",pybind11::overload_cast<const int8&, const long&>(&operator<))
    .def("__lt__",pybind11::overload_cast<const int8&, const double&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const int8&, const int16&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const int8&, const int32&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const int8&, const int64&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const int8&, const float32&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const int8&, const float64&>(&operator<))
    .def("__le__", &int8::operator<=)
    .def("__le__", pybind11::overload_cast<const int8&, const long&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const int8&, const double&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const int8&, const int16&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const int8&, const int32&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const int8&, const int64&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const int8&, const float32&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const int8&, const float64&>(&operator<=))
    .def("__gt__", &int8::operator>)
    .def("__gt__", pybind11::overload_cast<const int8&, const long&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const int8&, const double&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const int8&, const int16&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const int8&, const int32&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const int8&, const int64&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const int8&, const float32&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const int8&, const float64&>(&operator>))
    .def("__ge__", &int8::operator>=)
    .def("__ge__", pybind11::overload_cast<const int8&, const long&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const int8&, const double&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const int8&, const int16&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const int8&, const int32&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const int8&, const int64&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const int8&, const float32&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const int8&, const float64&>(&operator>=))
    .def("__lshift__", &int8::operator<<)
    .def("__lshift__", pybind11::overload_cast<const int8&, const long&>(&operator<<))
    .def("__lshift__", pybind11::overload_cast<const int8&, const int16&>(&operator<<))
    .def("__lshift__", pybind11::overload_cast<const int8&, const int32&>(&operator<<))
    .def("__lshift__", pybind11::overload_cast<const int8&, const int32&>(&operator<<))
    .def("__lshift__", pybind11::overload_cast<const int8&, const int64&>(&operator<<))
    .def("__rshift__", &int8::operator>>)
    .def("__rshift__", pybind11::overload_cast<const int8&, const long&>(&operator>>))
    .def("__rshift__", pybind11::overload_cast<const int8&, const int16&>(&operator>>))
    .def("__rshift__", pybind11::overload_cast<const int8&, const int32&>(&operator>>))
    .def("__rshift__", pybind11::overload_cast<const int8&, const int64&>(&operator>>))
    .def("__and__", &int8::operator&)
    .def("__and__", pybind11::overload_cast<const int8&, const long&>(&operator&))
    .def("__and__", pybind11::overload_cast<const int8&, const int16&>(&operator&))
    .def("__and__", pybind11::overload_cast<const int8&, const int32&>(&operator&))
    .def("__and__", pybind11::overload_cast<const int8&, const int64&>(&operator&))
    .def("__or__", &int8::operator|)
    .def("__or__", pybind11::overload_cast<const int8&, const long&>(&operator|))
    .def("__or__", pybind11::overload_cast<const int8&, const int16&>(&operator|))
    .def("__or__", pybind11::overload_cast<const int8&, const int32&>(&operator|))
    .def("__or__", pybind11::overload_cast<const int8&, const int64&>(&operator|))
    .def("__xor__", &int8::operator^)
    .def("__xor__", pybind11::overload_cast<const int8&, const long&>(&operator^))
    .def("__xor__", pybind11::overload_cast<const int8&, const int16&>(&operator^))
    .def("__xor__", pybind11::overload_cast<const int8&, const int32&>(&operator^))
    .def("__xor__", pybind11::overload_cast<const int8&, const int64&>(&operator^))
    .def("__invert__", &int8::operator~);

  pybind11::class_<int16>(m, "int16")
    .def(pybind11::init<int16_t>())
    .def("__repr__", &int16::repr)
    .def_property_readonly("nbits", &int16::nbits)
    .def("getValue", &int16::getValue)
    .def("__add__", &int16::operator+)
    .def("__add__", pybind11::overload_cast<const int16&, const long&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int16&, const double&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int16&, const std::complex<double>&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int16&, const int8&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int16&, const int32&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int16&, const int64&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int16&, const float32&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int16&, const float64&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int16&, const complex128&>(&operator+))
    .def("__sub__", &int16::operator-)
    .def("__sub__", pybind11::overload_cast<const int16&, const long&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const int16&, const double&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const int16&, const std::complex<double>&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const int16&, const int8&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const int16&, const int32&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const int16&, const int64&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const int16&, const float32&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const int16&, const float64&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const int16&, const complex128&>(&operator-))
    .def("__mul__", &int16::operator*)
    .def("__mul__", pybind11::overload_cast<const int16&, const long&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int16&, const double&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int16&, const std::complex<double>&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int16&, const int8&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int16&, const int8&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int16&, const int8&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int16&, const int8&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int16&, const int8&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int16&, const complex128&>(&operator*))
    .def("__truediv__", pybind11::overload_cast<const int16&, const int8&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int16&, const long&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int16&, const double&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int16&, const std::complex<double>&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int16&, const int8&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int16&, const int32&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int16&, const int64&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int16&, const float32&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int16&, const float64&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int16&, const complex128&>(&operator/))
    .def("__mod__", &int16::operator%)
    .def("__mod__", pybind11::overload_cast<const int16&, const long&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const int16&, const double&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const int16&, const int8&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const int16&, const int32&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const int16&, const int64&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const int16&, const float32&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const int16&, const float64&>(&operator%))
    .def("__floordiv__", &int16::floordiv)
    .def("__floordiv__", pybind11::overload_cast<const int16&, const long&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const int16&, const double&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const int16&, const int8&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const int16&, const int32&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const int16&, const int64&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const int16&, const float32&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const int16&, const float64&>(&floordiv))
    .def("__pow__", &int16::power)
    .def("__pow__", pybind11::overload_cast<const int16&, const long&>(&power))
    .def("__pow__", pybind11::overload_cast<const int16&, const double&>(&power))
    .def("__pow__", pybind11::overload_cast<const int16&, const std::complex<double>&>(&power))
    .def("__pow__", pybind11::overload_cast<const int16&, const int8&>(&power))
    .def("__pow__", pybind11::overload_cast<const int16&, const int32&>(&power))
    .def("__pow__", pybind11::overload_cast<const int16&, const int64&>(&power))
    .def("__pow__", pybind11::overload_cast<const int16&, const float64&>(&power))
    .def("__pow__", pybind11::overload_cast<const int16&, const float32&>(&power))
    .def("__pow__", pybind11::overload_cast<const int16&, const complex128&>(&power))
    .def("__abs__", &int16::abs)
    .def("__neg__", &int16::neg)
    .def("__pos__", &int16::pos)
    .def("__eq__", &int16::operator==)
    .def("__eq__", pybind11::overload_cast<const int16&, const long&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int16&, const double&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int16&, const std::complex<double>&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int16&, const int8&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int16&, const int32&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int16&, const int64&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int16&, const float32&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int16&, const float64&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int16&, const complex128&>(&operator==))
    .def("__ne__", &int16::operator!=)
    .def("__ne__", pybind11::overload_cast<const int16&, const long&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int16&, const double&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int16&, const std::complex<double>&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int16&, const int8&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int16&, const int32&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int16&, const int64&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int16&, const float32&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int16&, const float64&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int16&, const complex128&>(&operator!=))
    .def("__lt__", &int16::operator<)
    .def("__lt__", pybind11::overload_cast<const int16&, const long&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const int16&, const double&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const int16&, const int8&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const int16&, const int32&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const int16&, const int64&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const int16&, const float32&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const int16&, const float64&>(&operator<))
    .def("__le__", &int16::operator<=)
    .def("__le__", pybind11::overload_cast<const int16&, const long&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const int16&, const double&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const int16&, const int8&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const int16&, const int32&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const int16&, const int64&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const int16&, const float32&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const int16&, const float64&>(&operator<=))
    .def("__gt__", &int16::operator>)
    .def("__gt__", pybind11::overload_cast<const int16&, const long&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const int16&, const double&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const int16&, const int8&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const int16&, const int32&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const int16&, const int64&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const int16&, const float32&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const int16&, const float64&>(&operator>))
    .def("__ge__", &int16::operator>=)
    .def("__ge__", pybind11::overload_cast<const int16&, const long&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const int16&, const double&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const int16&, const int8&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const int16&, const int32&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const int16&, const int64&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const int16&, const float32&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const int16&, const float64&>(&operator>=))
    .def("__lshift__", &int16::operator<<)
    .def("__lshift__", pybind11::overload_cast<const int16&, const long&>(&operator<<))
    .def("__lshift__", pybind11::overload_cast<const int16&, const int8&>(&operator<<))
    .def("__lshift__", pybind11::overload_cast<const int16&, const int32&>(&operator<<))
    .def("__lshift__", pybind11::overload_cast<const int16&, const int64&>(&operator<<))
    .def("__rshift__", &int16::operator<<)
    .def("__rshift__", pybind11::overload_cast<const int16&, const long&>(&operator<<))
    .def("__rshift__", pybind11::overload_cast<const int16&, const int8&>(&operator<<))
    .def("__rshift__", pybind11::overload_cast<const int16&, const int32&>(&operator<<))
    .def("__rshift__", pybind11::overload_cast<const int16&, const int64&>(&operator<<))
    .def("__and__", &int16::operator&)
    .def("__and__", pybind11::overload_cast<const int16&, const long&>(&operator&))
    .def("__and__", pybind11::overload_cast<const int16&, const int8&>(&operator&))
    .def("__and__", pybind11::overload_cast<const int16&, const int32&>(&operator&))
    .def("__and__", pybind11::overload_cast<const int16&, const int64&>(&operator&))
    .def("__or__", &int16::operator|)
    .def("__or__", pybind11::overload_cast<const int16&, const long&>(&operator|))
    .def("__or__", pybind11::overload_cast<const int16&, const int8&>(&operator|))
    .def("__or__", pybind11::overload_cast<const int16&, const int32&>(&operator|))
    .def("__or__", pybind11::overload_cast<const int16&, const int64&>(&operator|))
    .def("__xor__", &int16::operator^)
    .def("__xor__", pybind11::overload_cast<const int16&, const long&>(&operator^))
    .def("__xor__", pybind11::overload_cast<const int16&, const int8&>(&operator^))
    .def("__xor__", pybind11::overload_cast<const int16&, const int32&>(&operator^))
    .def("__xor__", pybind11::overload_cast<const int16&, const int64&>(&operator^))
    .def("__invert__", &int16::operator~);

  pybind11::class_<int32>(m, "int32")
    .def(pybind11::init<int32_t>())
    .def("__repr__", &int32::repr)
    .def_property_readonly("nbits", &int32::nbits)
    .def("getValue", &int32::getValue)
    .def("__add__", &int32::operator+)
    .def("__add__", pybind11::overload_cast<const int32&, const long&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int32&, const double&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int32&, const std::complex<double>&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int32&, const int8&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int32&, const int16&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int32&, const int64&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int32&, const float32&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int32&, const float64&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int32&, const complex128&>(&operator+))
    .def("__sub__", &int32::operator-)
    .def("__sub__", pybind11::overload_cast<const int32&, const long&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const int32&, const double&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const int32&, const std::complex<double>&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const int32&, const int8&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const int32&, const int16&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const int32&, const int64&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const int32&, const float32&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const int32&, const float64&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const int32&, const complex128&>(&operator-))
    .def("__mul__", &int32::operator*)
    .def("__mul__", pybind11::overload_cast<const int32&, const long&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int32&, const double&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int32&, const std::complex<double>&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int32&, const int8&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int32&, const int16&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int32&, const int64&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int32&, const float32&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int32&, const float64&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int32&, const complex128&>(&operator*))
    .def("__truediv__", pybind11::overload_cast<const int32&, const long&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int32&, const double&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int32&, const std::complex<double>&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int32&, const int32&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int32&, const int8&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int32&, const int16&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int32&, const int64&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int32&, const float32&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int32&, const float64&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int32&, const complex128&>(&operator/))
    .def("__mod__", &int32::operator%)
    .def("__mod__", pybind11::overload_cast<const int32&, const long&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const int32&, const double&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const int32&, const int8&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const int32&, const int16&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const int32&, const int64&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const int32&, const float32&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const int32&, const float64&>(&operator%))
    .def("__floordiv__", &int32::floordiv)
    .def("__floordiv__", pybind11::overload_cast<const int32&, const long&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const int32&, const double&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const int32&, const int8&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const int32&, const int16&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const int32&, const int64&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const int32&, const float32&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const int32&, const float64&>(&floordiv))
    .def("__pow__", &int32::power)
    .def("__pow__", pybind11::overload_cast<const int32&, const long&>(&power))
    .def("__pow__", pybind11::overload_cast<const int32&, const double&>(&power))
    .def("__pow__", pybind11::overload_cast<const int32&, const std::complex<double>&>(&power))
    .def("__pow__", pybind11::overload_cast<const int32&, const int8&>(&power))
    .def("__pow__", pybind11::overload_cast<const int32&, const int16&>(&power))
    .def("__pow__", pybind11::overload_cast<const int32&, const int64&>(&power))
    .def("__pow__", pybind11::overload_cast<const int32&, const float32&>(&power))
    .def("__pow__", pybind11::overload_cast<const int32&, const float64&>(&power))
    .def("__pow__", pybind11::overload_cast<const int32&, const complex128&>(&power))
    .def("__abs__", &int32::abs)
    .def("__neg__", &int32::neg)
    .def("__pos__", &int32::pos)
    .def("__eq__", &int32::operator==)
    .def("__eq__", pybind11::overload_cast<const int32&, const long&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int32&, const double&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int32&, const std::complex<double>&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int32&, const int8&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int32&, const int16&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int32&, const int64&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int32&, const float32&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int32&, const float64&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int32&, const complex128&>(&operator==))
    .def("__ne__", &int32::operator!=)
    .def("__ne__", pybind11::overload_cast<const int32&, const long&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int32&, const double&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int32&, const std::complex<double>&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int32&, const int8&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int32&, const int16&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int32&, const int64&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int32&, const float32&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int32&, const float64&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int32&, const complex128&>(&operator!=))
    .def("__lt__", &int32::operator<)
    .def("__lt__", pybind11::overload_cast<const int32&, const long&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const int32&, const double&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const int32&, const int8&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const int32&, const int16&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const int32&, const int64&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const int32&, const float32&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const int32&, const float64&>(&operator<))
    .def("__le__", &int32::operator<=)
    .def("__le__", pybind11::overload_cast<const int32&, const long&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const int32&, const double&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const int32&, const int8&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const int32&, const int16&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const int32&, const int64&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const int32&, const float32&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const int32&, const float64&>(&operator<=))
    .def("__gt__", &int32::operator>)
    .def("__gt__", pybind11::overload_cast<const int32&, const long&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const int32&, const double&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const int32&, const int8&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const int32&, const int16&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const int32&, const int64&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const int32&, const float32&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const int32&, const float64&>(&operator>))
    .def("__ge__", &int32::operator>=)
    .def("__ge__", pybind11::overload_cast<const int32&, const long&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const int32&, const double&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const int32&, const int8&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const int32&, const int16&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const int32&, const int64&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const int32&, const float32&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const int32&, const float64&>(&operator>=))
    .def("__lshift__", &int32::operator<<)
    .def("__lshift__", pybind11::overload_cast<const int32&, const long&>(&operator<<))
    .def("__lshift__", pybind11::overload_cast<const int32&, const int8&>(&operator<<))
    .def("__lshift__", pybind11::overload_cast<const int32&, const int16&>(&operator<<))
    .def("__lshift__", pybind11::overload_cast<const int32&, const int64&>(&operator<<))
    .def("__rshift__", &int32::operator>>)
    .def("__rshift__", pybind11::overload_cast<const int32&, const long&>(&operator>>))
    .def("__rshift__", pybind11::overload_cast<const int32&, const int8&>(&operator>>))
    .def("__rshift__", pybind11::overload_cast<const int32&, const int16&>(&operator>>))
    .def("__rshift__", pybind11::overload_cast<const int32&, const int64&>(&operator>>))
    .def("__and__", &int32::operator&)
    .def("__and__", pybind11::overload_cast<const int32&, const long&>(&operator&))
    .def("__and__", pybind11::overload_cast<const int32&, const int8&>(&operator&))
    .def("__and__", pybind11::overload_cast<const int32&, const int16&>(&operator&))
    .def("__and__", pybind11::overload_cast<const int32&, const int64&>(&operator&))
    .def("__or__", &int32::operator|)
    .def("__or__", pybind11::overload_cast<const int32&, const long&>(&operator|))
    .def("__or__", pybind11::overload_cast<const int32&, const int8&>(&operator|))
    .def("__or__", pybind11::overload_cast<const int32&, const int16&>(&operator|))
    .def("__or__", pybind11::overload_cast<const int32&, const int64&>(&operator|))
    .def("__xor__", &int32::operator^)
    .def("__xor__", pybind11::overload_cast<const int32&, const long&>(&operator^))
    .def("__xor__", pybind11::overload_cast<const int32&, const int8&>(&operator^))
    .def("__xor__", pybind11::overload_cast<const int32&, const int16&>(&operator^))
    .def("__xor__", pybind11::overload_cast<const int32&, const int64&>(&operator^))
    .def("__invert__", &int32::operator~);

  pybind11::class_<int64>(m, "int64")
    .def(pybind11::init<int64_t>())
    .def("__repr__", &int64::repr)
    .def_property_readonly("nbits", &int64::nbits)
    .def("getValue", &int64::getValue)
    .def("__add__", &int64::operator+)
    .def("__add__", pybind11::overload_cast<const int64&, const long&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int64&, const double&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int64&, const std::complex<double>&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int64&, const int8&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int64&, const int16&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int64&, const int32&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int64&, const float32&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int64&, const float64&>(&operator+))
    .def("__add__", pybind11::overload_cast<const int64&, const complex128&>(&operator+))
    .def("__sub__", &int64::operator-)
    .def("'__sub__", pybind11::overload_cast<const int64&, const long&>(&operator-))
    .def("'__sub__", pybind11::overload_cast<const int64&, const double&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const int64&, const std::complex<double>&>(&operator-))
    .def("'__sub__", pybind11::overload_cast<const int64&, const int8&>(&operator-))
    .def("'__sub__", pybind11::overload_cast<const int64&, const int16&>(&operator-))
    .def("'__sub__", pybind11::overload_cast<const int64&, const int32&>(&operator-))
    .def("'__sub__", pybind11::overload_cast<const int64&, const float32&>(&operator-))
    .def("'__sub__", pybind11::overload_cast<const int64&, const float64&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const int64&, const complex128&>(&operator-))
    .def("__mul__", &int64::operator*)
    .def("__mul__", pybind11::overload_cast<const int64&, const long&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int64&, const double&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int64&, const std::complex<double>&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int64&, const int8&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int64&, const int16&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int64&, const int32&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int64&, const float32&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int64&, const float64&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const int64&, const complex128&>(&operator*))
    .def("__truediv__", pybind11::overload_cast<const int64&, const int64&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int64&, const long&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int64&, const double&>(&operator/))
    .def("__trueidiv__", pybind11::overload_cast<const int64&, const std::complex<double>&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int64&, const int8&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int64&, const int16&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int64&, const int32&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int64&, const float32&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int64&, const float64&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const int64&, const complex128&>(&operator/))
    .def("__mod__", &int64::operator%)
    .def("__mod__", pybind11::overload_cast<const int64&, const long&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const int64&, const double&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const int64&, const int8&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const int64&, const int16&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const int64&, const int32&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const int64&, const float32&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const int64&, const float64&>(&operator%))
    .def("__floordiv__", &int64::floordiv)
    .def("__floordiv__", pybind11::overload_cast<const int64&, const long&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const int64&, const double&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const int64&, const int8&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const int64&, const int16&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const int64&, const int32&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const int64&, const float32&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const int64&, const float64&>(&floordiv))
    .def("__pow__", &int64::power)
    .def("__pow__", pybind11::overload_cast<const int64&, const long&>(&power))
    .def("__pow__", pybind11::overload_cast<const int64&, const double&>(&power))
    .def("__pow__", pybind11::overload_cast<const int64&, const std::complex<double>&>(&power))
    .def("__pow__", pybind11::overload_cast<const int64&, const int8&>(&power))
    .def("__pow__", pybind11::overload_cast<const int64&, const int16&>(&power))
    .def("__pow__", pybind11::overload_cast<const int64&, const int32&>(&power))
    .def("__pow__", pybind11::overload_cast<const int64&, const float32&>(&power))
    .def("__pow__", pybind11::overload_cast<const int64&, const float64&>(&power))
    .def("__pow__", pybind11::overload_cast<const int64&, const complex128&>(&power))
    .def("__abs__", &int64::abs)
    .def("__neg__", &int64::neg)
    .def("__pos__", &int64::pos)
    .def("__eq__", &int64::operator==)
    .def("__eq__", pybind11::overload_cast<const int64&, const long&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int64&, const double&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int64&, const std::complex<double>&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int64&, const int8&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int64&, const int16&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int64&, const int32&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int64&, const float32&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int64&, const float64&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const int64&, const complex128&>(&operator==))
    .def("__ne__", &int64::operator!=)
    .def("__ne__", pybind11::overload_cast<const int64&, const long&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int64&, const double&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int64&, const std::complex<double>&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int64&, const int8&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int64&, const int16&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int64&, const int32&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int64&, const float32&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int64&, const float64&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const int64&, const complex128&>(&operator!=))
    .def("__lt__", &int64::operator<)
    .def("__lt__", pybind11::overload_cast<const int64&, const long&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const int64&, const double&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const int64&, const int8&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const int64&, const int16&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const int64&, const int32&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const int64&, const float32&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const int64&, const float64&>(&operator<))
    .def("__le__", &int64::operator<=)
    .def("__le__", pybind11::overload_cast<const int64&, const long&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const int64&, const double&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const int64&, const int8&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const int64&, const int16&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const int64&, const int32&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const int64&, const float32&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const int64&, const float64&>(&operator<=))
    .def("__gt__", &int64::operator>)
    .def("__gt__", pybind11::overload_cast<const int64&, const long&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const int64&, const double&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const int64&, const int8&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const int64&, const int16&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const int64&, const int32&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const int64&, const float32&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const int64&, const float64&>(&operator>))
    .def("__ge__", &int64::operator>=)
    .def("__ge__", pybind11::overload_cast<const int64&, const long&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const int64&, const double&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const int64&, const int8&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const int64&, const int16&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const int64&, const int32&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const int64&, const float32&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const int64&, const float64&>(&operator>=))
    .def("__lshift__", &int64::operator<<)
    .def("__lshift__", pybind11::overload_cast<const int64&, const long&>(&operator<<))
    .def("__lshift__", pybind11::overload_cast<const int64&, const int8&>(&operator<<))
    .def("__lshift__", pybind11::overload_cast<const int64&, const int16&>(&operator<<))
    .def("__lshift__", pybind11::overload_cast<const int64&, const int32&>(&operator<<))
    .def("__rshift__", &int64::operator>>)
    .def("__rshift__", pybind11::overload_cast<const int64&, const long&>(&operator>>))
    .def("__rshift__", pybind11::overload_cast<const int64&, const int8&>(&operator>>))
    .def("__rshift__", pybind11::overload_cast<const int64&, const int16&>(&operator>>))
    .def("__rshift__", pybind11::overload_cast<const int64&, const int32&>(&operator>>))
    .def("__and__", &int64::operator&)
    .def("__and__", pybind11::overload_cast<const int64&, const long&>(&operator&))
    .def("__and__", pybind11::overload_cast<const int64&, const int8&>(&operator&))
    .def("__and__", pybind11::overload_cast<const int64&, const int16&>(&operator&))
    .def("__and__", pybind11::overload_cast<const int64&, const int32&>(&operator&))
    .def("__or__", &int64::operator|)
    .def("__or__", pybind11::overload_cast<const int64&, const long&>(&operator|))
    .def("__or__", pybind11::overload_cast<const int64&, const int8&>(&operator|))
    .def("__or__", pybind11::overload_cast<const int64&, const int16&>(&operator|))
    .def("__or__", pybind11::overload_cast<const int64&, const int32&>(&operator|))
    .def("__xor__", &int64::operator^)
    .def("__xor__", pybind11::overload_cast<const int64&, const long&>(&operator^))
    .def("__xor__", pybind11::overload_cast<const int64&, const int8&>(&operator^))
    .def("__xor__", pybind11::overload_cast<const int64&, const int16&>(&operator^))
    .def("__xor__", pybind11::overload_cast<const int64&, const int32&>(&operator^))
    .def("__invert__", &int64::operator~);

  pybind11::class_<float32>(m, "float32")
    .def(pybind11::init<float>())
    .def(pybind11::init<int16_t>())
    .def(pybind11::init<int32_t>())
    .def(pybind11::init<int64_t>())
    .def("__repr__", &float32::repr)
    .def_property_readonly("nbits", &float32::nbits)
    .def("__add__", &float32::operator+)
    .def("__add__", pybind11::overload_cast<const float32&, const long&>(&operator+))
    .def("__add__", pybind11::overload_cast<const float32&, const double&>(&operator+))
    .def("__add__", pybind11::overload_cast<const float32&, const std::complex<double>&>(&operator+))
    .def("__add__", pybind11::overload_cast<const float32&, const int8&>(&operator+))
    .def("__add__", pybind11::overload_cast<const float32&, const int16&>(&operator+))
    .def("__add__", pybind11::overload_cast<const float32&, const int32&>(&operator+))
    .def("__add__", pybind11::overload_cast<const float32&, const int64&>(&operator+))
    .def("__add__", pybind11::overload_cast<const float32&, const float64&>(&operator+))
    .def("__add__", pybind11::overload_cast<const float32&, const complex128&>(&operator+))
    .def("__sub__", &float32::operator-)
    .def("__sub__", pybind11::overload_cast<const float32&, const long&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const float32&, const double&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const float32&, const std::complex<double>&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const float32&, const int8&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const float32&, const int16&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const float32&, const int32&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const float32&, const int64&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const float32&, const float64&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const float32&, const complex128&>(&operator-))
    .def("__mul__", &float32::operator*)
    .def("__mul__", pybind11::overload_cast<const float32&, const long&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const float32&, const double&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const float32&, const std::complex<double>&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const float32&, const int8&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const float32&, const int16&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const float32&, const int32&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const float32&, const int64&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const float32&, const float64&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const float32&, const complex128&>(&operator*))
    .def("__truediv__", &float32::operator/)
    .def("__truediv__", pybind11::overload_cast<const float32&, const long&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const float32&, const double&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const float32&, const std::complex<double>&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const float32&, const int8&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const float32&, const int16&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const float32&, const int32&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const float32&, const int64&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const float32&, const float64&>(&operator/))
    .def("__treudiv__", pybind11::overload_cast<const float32&, const complex128&>(&operator/))
    .def("__mod__", &float32::operator%)
    .def("__mod__", pybind11::overload_cast<const float32&, const long&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const float32&, const double&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const float32&, const int8&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const float32&, const int16&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const float32&, const int32&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const float32&, const int64&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const float32&, const float64&>(&operator%))
    .def("__floordiv__", &float32::floordiv)
    .def("__floordiv__", pybind11::overload_cast<const float32&, const long&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const float32&, const double&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const float32&, const int8&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const float32&, const int16&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const float32&, const int32&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const float32&, const int64&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const float32&, const float64&>(&floordiv))
    .def("__pow__", &float32::power)
    .def("__pow__", pybind11::overload_cast<const float32&, const long&>(&power))
    .def("__pow__", pybind11::overload_cast<const float32&, const double&>(&power))
    .def("__pow__", pybind11::overload_cast<const float32&, const std::complex<double>&>(&power))
    .def("__pow__", pybind11::overload_cast<const float32&, const int8&>(&power))
    .def("__pow__", pybind11::overload_cast<const float32&, const int16&>(&power))
    .def("__pow__", pybind11::overload_cast<const float32&, const int32&>(&power))
    .def("__pow__", pybind11::overload_cast<const float32&, const int64&>(&power))
    .def("__pow__", pybind11::overload_cast<const float32&, const float64&>(&power))
    .def("__pow__", pybind11::overload_cast<const float32&, const complex128&>(&power))
    .def("__abs__", &float32::abs)
    .def("__neg__", &float32::neg)
    .def("__pos__", &float32::pos)
    .def("__eq__", &float32::operator==)
    .def("__eq__", pybind11::overload_cast<const float32&, const long&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const float32&, const double&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const float32&, const std::complex<double>&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const float32&, const int8&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const float32&, const int16&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const float32&, const int32&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const float32&, const int64&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const float32&, const float64&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const float32&, const complex128&>(&operator==))
    .def("__ne__", &float32::operator!=)
    .def("__ne__", pybind11::overload_cast<const float32&, const long&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const float32&, const double&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const float32&, const std::complex<double>&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const float32&, const int8&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const float32&, const int16&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const float32&, const int32&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const float32&, const int64&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const float32&, const float64&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const float32&, const complex128&>(&operator!=))
    .def("__lt__", &float32::operator<)
    .def("__lt__", pybind11::overload_cast<const float32&, const long&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const float32&, const double&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const float32&, const int8&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const float32&, const int16&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const float32&, const int32&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const float32&, const int64&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const float32&, const float64&>(&operator<))
    .def("__le__", &float32::operator<=)
    .def("__le__", pybind11::overload_cast<const float32&, const long&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const float32&, const double&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const float32&, const int8&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const float32&, const int16&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const float32&, const int32&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const float32&, const int64&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const float32&, const float64&>(&operator<=))
    .def("__gt__", &float32::operator>)
    .def("__gt__", pybind11::overload_cast<const float32&, const long&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const float32&, const double&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const float32&, const int8&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const float32&, const int16&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const float32&, const int32&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const float32&, const int64&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const float32&, const float64&>(&operator>))
    .def("__ge__", &float32::operator>=)
    .def("__ge__", pybind11::overload_cast<const float32&, const long&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const float32&, const double&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const float32&, const int8&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const float32&, const int16&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const float32&, const int32&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const float32&, const int64&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const float32&, const float64&>(&operator>=));

  pybind11::class_<float64>(m, "float64")
    .def(pybind11::init<float>())
    .def(pybind11::init<int16_t>())
    .def(pybind11::init<int32_t>())
    .def(pybind11::init<int64_t>())
    .def("__repr__", &float64::repr)
    .def_property_readonly("nbits", &float64::nbits)
    .def("__add__", &float64::operator+)
    .def("__add__", pybind11::overload_cast<const float64&, const long&>(&operator+))
    .def("__add__", pybind11::overload_cast<const float64&, const double&>(&operator+))
    .def("__add__", pybind11::overload_cast<const float64&, const std::complex<double>&>(&operator+))
    .def("__add__", pybind11::overload_cast<const float64&, const int8&>(&operator+))
    .def("__add__", pybind11::overload_cast<const float64&, const int16&>(&operator+))
    .def("__add__", pybind11::overload_cast<const float64&, const int32&>(&operator+))
    .def("__add__", pybind11::overload_cast<const float64&, const int64&>(&operator+))
    .def("__add__", pybind11::overload_cast<const float64&, const float32&>(&operator+))
    .def("__add__", pybind11::overload_cast<const float64&, const complex128&>(&operator+))
    .def("__sub__", &float64::operator-)
    .def("__sub__", pybind11::overload_cast<const float64&, const long&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const float64&, const double&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const float64&, const std::complex<double>&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const float64&, const int8&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const float64&, const int16&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const float64&, const int32&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const float64&, const int64&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const float64&, const float32&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const float64&, const complex128&>(&operator-))
    .def("__mul__", &float64::operator*)
    .def("__mul__", pybind11::overload_cast<const float64&, const long&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const float64&, const double&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const float64&, const std::complex<double>&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const float64&, const int8&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const float64&, const int16&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const float64&, const int32&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const float64&, const int64&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const float64&, const float32&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const float64&, const complex128&>(&operator*))
    .def("__truediv__", &float64::operator/)
    .def("__truediv__", pybind11::overload_cast<const float64&, const long&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const float64&, const double&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const float64&, const std::complex<double>&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const float64&, const int8&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const float64&, const int16&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const float64&, const int32&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const float64&, const int64&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const float64&, const float32&>(&operator/))
    .def("__ttrueidiv__", pybind11::overload_cast<const float64&, const complex128&>(&operator/))
    .def("__mod__", &float64::operator%)
    .def("__mod__", pybind11::overload_cast<const float64&, const long&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const float64&, const double&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const float64&, const int8&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const float64&, const int16&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const float64&, const int32&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const float64&, const int64&>(&operator%))
    .def("__mod__", pybind11::overload_cast<const float64&, const float32&>(&operator%))
    .def("__floordiv__", &float64::floordiv)
    .def("__floordiv__", pybind11::overload_cast<const float64&, const long&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const float64&, const double&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const float64&, const int8&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const float64&, const int16&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const float64&, const int32&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const float64&, const int64&>(&floordiv))
    .def("__floordiv__", pybind11::overload_cast<const float64&, const float32&>(&floordiv))
    .def("__pow__", &float64::power)
    .def("__pow__", pybind11::overload_cast<const float64&, const long&>(&power))
    .def("__pow__", pybind11::overload_cast<const float64&, const double&>(&power))
    .def("__pow__", pybind11::overload_cast<const float64&, const std::complex<double>&>(&power))
    .def("__pow__", pybind11::overload_cast<const float64&, const int8&>(&power))
    .def("__pow__", pybind11::overload_cast<const float64&, const int16&>(&power))
    .def("__pow__", pybind11::overload_cast<const float64&, const int32&>(&power))
    .def("__pow__", pybind11::overload_cast<const float64&, const int64&>(&power))
    .def("__pow__", pybind11::overload_cast<const float64&, const float32&>(&power))
    .def("__pow__", pybind11::overload_cast<const float64&, const complex128&>(&power))
    .def("__abs__", &float64::abs)
    .def("__neg__", &float64::neg)
    .def("__pos__", &float64::pos)
    .def("__eq__", &float64::operator==)
    .def("__eq__", pybind11::overload_cast<const float64&, const long&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const float64&, const double&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const float64&, const std::complex<double>&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const float64&, const int8&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const float64&, const int16&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const float64&, const int32&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const float64&, const int64&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const float64&, const float32&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const float64&, const complex128&>(&operator==))
    .def("__ne__", &float64::operator!=)
    .def("__ne__", pybind11::overload_cast<const float64&, const long&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const float64&, const double&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const float64&, const std::complex<double>&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const float64&, const int8&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const float64&, const int16&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const float64&, const int32&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const float64&, const int64&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const float64&, const float32&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const float64&, const complex128&>(&operator!=))
    .def("__lt__", &float64::operator<)
    .def("__lt__", pybind11::overload_cast<const float64&, const long&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const float64&, const double&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const float64&, const int8&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const float64&, const int16&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const float64&, const int32&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const float64&, const int64&>(&operator<))
    .def("__lt__", pybind11::overload_cast<const float64&, const float32&>(&operator<))
    .def("__le__", &float64::operator<=)
    .def("__le__", pybind11::overload_cast<const float64&, const long&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const float64&, const double&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const float64&, const int8&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const float64&, const int16&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const float64&, const int32&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const float64&, const int64&>(&operator<=))
    .def("__le__", pybind11::overload_cast<const float64&, const float32&>(&operator<=))
    .def("__gt__", &float64::operator>)
    .def("__gt__", pybind11::overload_cast<const float64&, const long&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const float64&, const double&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const float64&, const int8&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const float64&, const int16&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const float64&, const int32&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const float64&, const int64&>(&operator>))
    .def("__gt__", pybind11::overload_cast<const float64&, const float32&>(&operator>))
    .def("__ge__", &float64::operator>=)
    .def("__ge__", pybind11::overload_cast<const float64&, const long&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const float64&, const double&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const float64&, const int8&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const float64&, const int16&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const float64&, const int32&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const float64&, const int64&>(&operator>=))
    .def("__ge__", pybind11::overload_cast<const float64&, const float32&>(&operator>=));

  pybind11::class_<complex128>(m, "complex128")
    .def(pybind11::init<double,double>())
    .def(pybind11::init<float64,float64>())
    .def(pybind11::init<std::complex<double>>())
    .def_property_readonly("nbits", &complex128::nbits)
    .def_property_readonly("real", &complex128::real)
    .def_property_readonly("imag", &complex128::imag)
    .def("__repr__", &complex128::repr)
    .def("value", &complex128::getValue)
    .def("conj", &complex128::conj)
    .def("magnitude", &complex128::magnitude)
    .def("phase", &complex128::phase)
    .def("__add__", &complex128::operator+)
    .def("__add__", pybind11::overload_cast<const complex128&, const long&>(&operator+))
    .def("__add__", pybind11::overload_cast<const complex128&, const double&>(&operator+))
    .def("__add__", pybind11::overload_cast<const complex128&, const std::complex<double>&>(&operator+))
    .def("__add__", pybind11::overload_cast<const complex128&, const int8&>(&operator+))
    .def("__add__", pybind11::overload_cast<const complex128&, const int16&>(&operator+))
    .def("__add__", pybind11::overload_cast<const complex128&, const int32&>(&operator+))
    .def("__add__", pybind11::overload_cast<const complex128&, const int64&>(&operator+))
    .def("__add__", pybind11::overload_cast<const complex128&, const float32&>(&operator+))
    .def("__add__", pybind11::overload_cast<const complex128&, const float64&>(&operator+))
    .def("__sub__", &complex128::operator-)
    .def("__sub__", pybind11::overload_cast<const complex128&, const long&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const complex128&, const double&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const complex128&, const std::complex<double>&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const complex128&, const int8&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const complex128&, const int16&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const complex128&, const int32&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const complex128&, const int64&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const complex128&, const float32&>(&operator-))
    .def("__sub__", pybind11::overload_cast<const complex128&, const float64&>(&operator-))
    .def("__mul__", &complex128::operator*)
    .def("__mul__", pybind11::overload_cast<const complex128&, const long&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const complex128&, const double&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const complex128&, const std::complex<double>&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const complex128&, const int8&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const complex128&, const int16&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const complex128&, const int32&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const complex128&, const int64&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const complex128&, const float32&>(&operator*))
    .def("__mul__", pybind11::overload_cast<const complex128&, const float64&>(&operator*))
    .def("__truediv__", &complex128::operator/)
    .def("__truediv__", pybind11::overload_cast<const complex128&, const long&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const complex128&, const double&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const complex128&, const std::complex<double>&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const complex128&, const int8&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const complex128&, const int16&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const complex128&, const int32&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const complex128&, const int64&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const complex128&, const float32&>(&operator/))
    .def("__truediv__", pybind11::overload_cast<const complex128&, const float64&>(&operator/))
    .def("__pow__", &complex128::power)
    .def("__pow__", pybind11::overload_cast<const complex128&, const long&>(&power))
    .def("__pow__", pybind11::overload_cast<const complex128&, const double&>(&power))
    .def("__pow__", pybind11::overload_cast<const complex128&, const std::complex<double>&>(&power))
    .def("__pow__", pybind11::overload_cast<const complex128&, const int8&>(&power))
    .def("__pow__", pybind11::overload_cast<const complex128&, const int16&>(&power))
    .def("__pow__", pybind11::overload_cast<const complex128&, const int32&>(&power))
    .def("__pow__", pybind11::overload_cast<const complex128&, const int64&>(&power))
    .def("__pow__", pybind11::overload_cast<const complex128&, const float32&>(&power))
    .def("__pow__", pybind11::overload_cast<const complex128&, const float64&>(&power))
    .def("__eq__", &complex128::operator==)
    .def("__eq__", pybind11::overload_cast<const complex128&, const long&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const complex128&, const double&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const complex128&, const std::complex<double>&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const complex128&, const int8&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const complex128&, const int16&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const complex128&, const int32&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const complex128&, const int64&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const complex128&, const float32&>(&operator==))
    .def("__eq__", pybind11::overload_cast<const complex128&, const float64&>(&operator==))
    .def("__ne__", &complex128::operator!=)
    .def("__ne__", pybind11::overload_cast<const complex128&, const long&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const complex128&, const double&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const complex128&, const std::complex<double>&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const complex128&, const int8&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const complex128&, const int16&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const complex128&, const int32&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const complex128&, const int64&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const complex128&, const float32&>(&operator!=))
    .def("__ne__", pybind11::overload_cast<const complex128&, const float64&>(&operator!=));
}