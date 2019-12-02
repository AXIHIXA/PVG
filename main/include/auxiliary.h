//
// Created by ax on 11/30/19.
//

#ifndef PVG_AUXILIARY_H
#define PVG_AUXILIARY_H

#include <boost/lexical_cast.hpp>
#include <string>
#include <vector>

namespace pvgaux
{

template < typename S, typename T >
inline T to_num(const S & x)
{
    return boost::lexical_cast< T, S >(x);
}

template < typename T >
std::vector< T > split_to_num(const char * s, char c = ' ')
{
    if (!s)
    {
        return std::vector< T >();
    }

    std::string s_(s);
    std::vector< std::string > str;

    unsigned pos = 0;

    while (pos < s_.size())
    {
        while (pos < s_.size() && s_[pos] == c)
        {
            ++pos;
        }

        unsigned end = pos;

        do
        {
            ++end;
        }
        while (end < s_.size() && s_[end] != c);

        if (pos < s_.size())
        {
            str.push_back(s_.substr(pos, end - pos));
        }

        pos = end;
    }

    std::vector< T > num(str.size());

    for (size_t i = 0; i < str.size(); ++i)
    {
        num[i] = to_num< std::string, T >(str[i]);
    }

    return num;
}

}


#endif // PVG_AUXILIARY_H
