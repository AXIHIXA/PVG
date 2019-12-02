#ifndef AUXILIARY_H
#define AUXILIARY_H

#include <opencv2/core.hpp>
#include <boost/lexical_cast.hpp>
#include <vector_types.h>
#include "structure.h"

class Region;

void cuda_opencv_initialization(int w, int h);
bool is_image_boundary(const Region& region, int ring, int row, int col);

void difference(const cv::Mat& img1, const cv::Mat& groundtruth, const cv::Rect& rect = cv::Rect());

template<typename S, typename T>
inline T to_num(const S& x)
{
	return boost::lexical_cast<T,S>(x);
}

template<typename T>
std::vector<T> split_to_num(const char* s, char c=' ')
{
	if (s==nullptr) return std::vector<T>();
	std::string s_(s);
	std::vector<std::string> str;
	unsigned pos = 0;
	while (pos<s_.size())
	{
		while (pos<s_.size() && s_[pos] == c)
		{
			++pos;
		}
		unsigned end = pos;

		do{
			++end;
		} while (end<s_.size() && s_[end] != c);

		if (pos<s_.size()) str.push_back(s_.substr(pos, end - pos));
		pos = end;
	}

	std::vector<T> num(str.size());
	for (size_t i = 0; i < str.size(); ++i)
	{
		num[i] = to_num<string, T>(str[i]);
	}
	return num;
}

template<typename T>
std::vector<T> union_set(std::vector<T>& s1, const std::vector<T>& s2)
{
	std::vector<T> tmp;
	for(size_t i=0;i<s2.size();++i)
	{
		bool existed=false;
		for(size_t j=0;j<s1.size();++j)
		{
			if(s1[j]==s2[i])
			{
				existed=true;
				break;
			}
		}
		if (!existed) {
			s1.push_back(s2[i]);
			tmp.push_back(s2[i]);
		}
	}
	return tmp;
};

double green_integral(const float2& func, double x1, double x2, double y1, double y2);
void save_basis(const CPoint2d& pt, const std::vector<int>& index, const std::vector<float>& coefs, const std::vector<TreeNodeD>& nodes, const std::vector<CPoint2d>& knots);
void save_mesh(const cv::Mat& image);

#endif
