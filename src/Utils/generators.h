/**
 * Copyright 2020 Guillaume AUJAY. All rights reserved.
 *
 */

#ifndef GENERATORS_H
#define GENERATORS_H

#include <cstdlib>
#include <ctime>
#include <cmath>
// #include <stdlib.h>

#include <vector>
#include <string>
#include <numeric>
#include <iostream>
#include <algorithm>

/****************************************************************************************************/

template <const size_t N>
struct data8 {
  int8_t d[N];
};

template <const size_t N>
struct data16 {
  int16_t d[N];
};

template <const size_t N>
struct data32 {
  int32_t d[N];
};

template <const size_t N>
struct data64 {
  int64_t d[N];
  inline data64& operator++()
  { ++(d[0]); return *this; }
};

/****************************************************************************************************/

template <typename T>
inline T* allocate(size_t N)
{
  return static_cast<T*>(malloc(sizeof(T)*N));
}

/****************************************************************************************************/

template <typename T>
void print_vec(const std::vector<T>& v, const std::string& name="vec", bool graph=false)
{
  std::cout << name << ": {";
  
  for (int i=0; i<v.size()-1; ++i)
    std::cout << v[i] << ", ";
	
  if (v.size() > 0)
	std::cout << v[v.size()-1];

  std::cout << "}\n";
  
  // Graph
  if (graph) {
    for (auto a : v)
    {
      std::cout << " |";
      for (int i=0; i<a; ++i)
        std::cout << "-";
      std::cout << a << "\n";
    }
  }
}

template <typename T>
void print_vec(const T* v, const std::string& name="vec", int n=8, bool graph=false)
{
  std::cout << name << ": {";
  
  for (int i=0; i<n-1; ++i)
    std::cout << (int)*(v+i) << ", ";
	
  if (n > 0)
    std::cout << (int)*(v+n-1);

  std::cout << "}\n";
  
  // Graph
  if (graph) {
    for (int j=0; j<n; ++j)
    {
      auto a = *(v+j);
      std::cout << " |";
      for (int i=0; i<a; ++i)
        std::cout << "-";
      std::cout << a << "\n";
    }
  }
}

/****************************************************************************************************/

template <typename T>
inline T get_one()
{
  return T(); //default constructed
}

template <typename T>
inline void vec_val(std::vector<T>& v, T value=0)
{
  std::fill(std::begin(v), std::end(v), value);
}

template <typename T>
inline void vec_seq(T* v, size_t N, T start)
{
  for (size_t i=0; i<N; ++i)
    v[i] = start++;
}

template <typename T>
inline void vec_seq(std::vector<T>& v, T start=0)
{
  std::iota(std::begin(v), std::end(v), start);
}

template <typename T>
inline void vec_inv(T* v, size_t N, T start)
{
  for (int i=N-1; i>=0; --i)
    v[i] = start++;
}

template <typename T>
inline void vec_inv(std::vector<T>& v, T start=0)
{
  for (int i=v.size()-1; i>=0; --i)
    v[i] = start++;
}

template <typename T>
inline void vec_eod(std::vector<T>& v, T start=0)
{
  const size_t N = v.size();
  for (int i=0; i<(N+1)/2; ++i)
    v[i] = start + i;

  for (int i=(N+1)/2; i<N; ++i)
    v[i] = start++;
}

template <typename T>
inline void vec_evn(std::vector<T>& v, T start=0)
{
  const size_t N = v.size();
  for (int i=0; i<N-1; i+=2)
  {
    v[i  ] = start + i/2;
	v[i+1] = start + N-1 - i/2;
  }
  v[N-1] = start + N/2;
}

template <typename T>
inline void vec_odd(std::vector<T>& v, T start=0)
{
  const size_t N = v.size();
  for (int i=0; i<N; i+=2)
    v[i] = start + i;
	
  for (int i=1; i<N; i+=2)
    v[i] = start + N-1 - i;
}

template <typename T>
inline void vec_pbk(std::vector<T>& v, T start=0)
{
  const size_t N = v.size();
  v[N-1] = start++;
  
  for (int i=0; i<N-1; ++i)
    v[i] = start++;
}

template <typename T>
inline void vec_pft(std::vector<T>& v, T start=0)
{
  for (int i=1; i<v.size(); ++i)
    v[i] = start++;
	
  v[0] = start;
}

template <typename T>
inline void vec_pmd(std::vector<T>& v, T start=0)
{
  const size_t N = v.size();
  v[N/2] = start;
  
  for (int i=0; i<N/2; ++i)
    v[i] = ++start;
	
  for (int i=(N/2)+1; i<N; ++i)
    v[i] = ++start;
}

template <typename T>
inline void vec_pip(std::vector<T>& v, T start=0)
{
  const size_t N = v.size();
  for (int i=0; i<N/2; ++i)
    v[i] = start+i;
  
  for (int i=N-1; i>=N/2; --i)
    v[i] = start++;
}

template <typename T>
inline void vec_ipp(std::vector<T>& v, T start=0)
{
  const size_t N = v.size();

  T c = start;
  for (int i=(N-1)/2; i>=0; --i)
    v[i] = c++;
  
  c = start;
  for (int i=N/2; i<N; ++i)
    v[i] = c++;
}

template <typename T>
inline void vec_rnd(std::vector<T>& v)
{
  std::generate(v.begin(), v.end(), []() { return std::rand(); });
}

template <typename T>
inline void vec_rrd(T* v, size_t N, T max)
{
  for (size_t i=0; i<N; ++i)
    v[i] = (T)(std::round(max * std::rand()/(float)RAND_MAX));
}

template <typename T>
inline void vec_rrd(std::vector<T>& v, T max)
{
  std::generate(v.begin(), v.end(), [max]() {
      return (T)(std::round(max * std::rand()/(float)RAND_MAX));
    });
}

template <typename T>
inline void vec_rrd(T* v, size_t N, T min, T max)
{
  for (size_t i=0; i<N; ++i)
    v[i] = min + (T)(std::round((max-min) * std::rand()/(float)RAND_MAX));
}

template <typename T>
inline void vec_rrd(std::vector<T>& v, T min, T max)
{
  std::generate(v.begin(), v.end(), [min, max]() {
      return min + (T)(std::round((max-min) * std::rand()/(float)RAND_MAX));
    });
}

template <typename T>
inline void vec_rrdf(T* v, size_t N, T min, T max)
{
  for (size_t i=0; i<N; ++i)
    v[i] = min + (max-min) * std::rand()/(T)RAND_MAX;
}

template <typename T>
inline void vec_rrdf(std::vector<T>& v, T min, T max)
{
  std::generate(v.begin(), v.end(), [min, max]() {
      return min + (max-min) * std::rand()/(T)RAND_MAX;
    });
}

template <typename T1, typename T2>
struct DualVector {
  DualVector(size_t N) : u(N), v(N) {}
  
  std::vector<T1> u;
  std::vector<T2> v;
};

template <typename T1, typename T2>
inline std::vector<DualVector<T1,T2>> dual_vec_rrd(size_t N, size_t M, int min, int max)
{
  std::vector<DualVector<T1,T2>> vec;
  vec.reserve(N);
  for (size_t i=0; i<N; ++i)
  {
    DualVector<T1,T2> dv(M);
    vec_rrd(dv.u, (T1)min, (T1)max);
    vec_rrd(dv.v, (T2)min, (T2)max);
    
    vec.push_back(dv);
  }
  return vec;
}

template <typename T>
inline std::vector<DualVector<T,T>> dual_vec_rrdf(size_t N, size_t M, T min, T max)
{
  std::vector<DualVector<T,T>> vec;
  vec.reserve(N);
  for (size_t i=0; i<N; ++i)
  {
    DualVector<T,T> dv(M);
    vec_rrdf(dv.u, min, max);
    vec_rrdf(dv.v, min, max);
    
    vec.push_back(dv);
  }
  return vec;
}

template <typename T>
inline void vec_shf(std::vector<T>& v, float proba=0.01f)
{
  const size_t N = v.size();
  proba *= RAND_MAX;
  
  for (int i=0; i<N; ++i)
  {
    if (std::rand() > proba)
      v[i] = i;
    else
      v[i] = std::round((N-1) * (std::rand()/(float)RAND_MAX));
  }
}

/****************************************************************************************************/

inline char get_rand_printable_char(unsigned int seed)
{
  std::srand(seed);
  return (char)(33 + 93.f*(std::rand()/(float)RAND_MAX));
}

template <typename T>
inline T get_rand_unit()
{
	return (std::rand()/(T)RAND_MAX);
}

template <typename T>
inline T get_rand(T min, T max)
{
	return (std::rand()/(double)RAND_MAX)*(max-min) - min;
}

template <typename T>
inline T get_rand(unsigned int seed=0)
{
  std::srand(seed);
  return (T)(std::rand());  //forced conversion
}

template <>
inline char get_rand<char>(unsigned int seed)
{
  std::srand(seed);
  return (char)(-128 + 255.f*(std::rand()/(float)RAND_MAX));
}

template <>
inline std::string get_rand<std::string>(unsigned int len)
{
  return std::string(len, get_rand_printable_char(len));  //length as seed
}

/****************************************************************************************************/

template <typename T>
void make_random_data(size_t N, T& a, unsigned int seed=1)
{
  std::srand(seed);
  // unsigned int s1 = seed, s2 = seed;
  for (size_t i=0; i<N; ++i) {
    // a[i] = rand_r(&s1)*rand_r(&s2);  //Linux
    a[i] = std::rand();
  }
}

/****************************************************************************************************/

// static std::string make_random_string(size_t len, bool printable=true, unsigned int seed=1)
// {
  // char tmp[len+1];
  // char offset = 1; float mult = 254.f;
  // if (printable) {
    // offset = 33; mult = 93.f;
  // }
  
  // std::srand(seed);
  // for (size_t i=0; i<len; ++i) {
    // tmp[i] = offset + mult*(std::rand()/(float)RAND_MAX);
  // }
  // tmp[len] = '\0';
  
  // return std::string(tmp);
// }

/****************************************************************************************************/

template <typename T>
static std::vector<T> createFixStringVector(const size_t n, const size_t len)
{
  std::vector<T> v;
  v.reserve(n);
  char tmp[len+1];
  std::srand(0); //std::time(nullptr));
  
  for (size_t i=0; i<n; ++i)
  {
    for (size_t j=0; j<len; ++j) {
      tmp[j] = 33 + (char)(93.f*(std::rand()/(float)RAND_MAX));
    }
    tmp[len] = '\0';
    v.emplace_back((char*)&tmp);
    // if (v.back().size() != len) std::cout << "Fail Fix: " << v.back().size() << "|" << len << std::endl;
  }

  return  v;
}

/****************************************************************************************************/

template <typename T>
static std::vector<T> createStringVector(const size_t n, const size_t maxLen, const size_t minLen=0)
{
  std::vector<T> v;
  if (maxLen<minLen) return v;
  v.reserve(n);
  char tmp[maxLen+1];
  std::srand(0); //std::time(nullptr));
  
  for (size_t i=0; i<n; ++i)
  {
    const size_t len = minLen + (size_t)((float)(maxLen-minLen)*(std::rand()/(float)RAND_MAX));
    
    for (size_t j=0; j<len; ++j) {
      tmp[j] = 33 + (char)(93.f*(std::rand()/(float)RAND_MAX)); //printable char
    }
    tmp[len] = '\0';
    v.emplace_back((char*)&tmp);
    // if (v.back().size() != len) std::cout << "Fail: " << v.back().size() << "|" << len << std::endl;
  }

  return  v;
}

/****************************************************************************************************/

#endif // GENERATORS_H
