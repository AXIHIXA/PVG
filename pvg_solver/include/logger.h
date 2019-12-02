#ifndef LOGGER_H
#define LOGGER_H

#include <string>

class Logger
{
private:
	std::string contents;
	Logger();
public:
	~Logger();
	static Logger& ins();
	friend Logger& operator<<(Logger &s, const char* p);
	friend Logger& operator<<(Logger &s, const std::string& p);
	friend Logger& operator<<(Logger &s, const signed int p);
	friend Logger& operator<<(Logger &s, const unsigned int p);
	friend Logger& operator<<(Logger &s, const double p);
	friend Logger& operator<<(Logger &s, const float p);
	friend Logger& operator<<(Logger &s, const std::size_t p);
};

#endif
