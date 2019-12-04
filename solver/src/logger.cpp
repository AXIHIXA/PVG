#include "logger.h"
#include "auxiliary.h"
#include <fstream>
#include <memory>

using namespace std;

string base_dir = "./resultImg/";

Logger::Logger()
{
}

Logger::~Logger()
{
	ofstream ofile(base_dir + "logger");
	ofile << contents;
	ofile.close();
}

Logger& Logger::ins()
{
	static unique_ptr<Logger> log_obj(new Logger());
	return *log_obj;
}

Logger& operator<<(Logger &s, const char* p)
{
	s.contents += p;
	return s;
}

Logger& operator<<(Logger &s, const std::string& p)
{
	s.contents += p;
	return s;
}

Logger& operator<<(Logger &s, const signed int p)
{
	s.contents += to_num<int, string>(p);
	return s;
}

Logger& operator<<(Logger &s, const unsigned int p)
{
	s.contents += to_num<unsigned int, string>(p);
	return s;
}

Logger& operator<<(Logger &s, const double p)
{
	s.contents += to_num<double, string>(p);
	return s;
}

Logger& operator<<(Logger &s, const float p)
{
	s.contents += to_num<float, string>(p);
	return s;
}

#ifdef _WIN64
Logger& operator<<(Logger &s, const std::size_t p)
{
	s.contents += to_num<size_t, string>(p);
	return s;
}
#endif
