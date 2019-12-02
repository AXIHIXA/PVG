#ifndef DEBUG_H
#define DEBUG_H

#include <stdio.h>

#define ENDL "\n"

#ifdef ST_COLOR
#define KNRM "\033[0m"
#define KRED "\033[1;31m"
#define KGRN "\033[1;32m"
#define KYEL "\033[1;33m"
#define KBLU "\033[1;34m"
#define KMAG "\033[1;35m"
#define KCYN "\033[1;36m"
#define KWHT "\033[1;37m"
#define KBWN "\033[0;33m"
#else
#define KNRM ""
#define KRED ""
#define KGRN ""
#define KYEL ""
#define KBLU ""
#define KMAG ""
#define KCYN ""
#define KWHT ""
#define KBWN ""
#endif

#ifdef ST_VERBOSE
#define ST_DEBUG
#define ST_INFO
#define ST_WARN
#define ST_ERROR
#define ST_SUCCESS
#endif

#ifdef ST_DEBUG
#define st_debug(S, ...)                                                           \
    do                                                                             \
    {                                                                              \
        fprintf(stderr, KMAG "DEBUG: %s %s:%d " KNRM S ENDL,                       \
                __extension__ __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__);    \
    }                                                                              \
    while (0)
#else
#define st_debug(S, ...)
#endif

#ifdef ST_INFO
#define st_info(S, ...)                                                            \
    do                                                                             \
    {                                                                              \
        fprintf(stderr, KBLU "INFO: %s %s:%d " KNRM S ENDL,                        \
                __extension__ __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__);    \
    }                                                                              \
    while (0)
#else
#define st_info(S, ...)
#endif

#ifdef ST_WARN
#define st_warn(S, ...)                                                            \
    do                                                                             \
    {                                                                              \
        fprintf(stderr, KYEL "WARN: %s %s:%d " KNRM S ENDL,                        \
                __extension__ __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__);    \
    }                                                                              \
    while (0)
#else
#define st_warn(S, ...)
#endif

#ifdef ST_SUCCESS
#define st_success(S, ...)                                                         \
    do                                                                             \
    {                                                                              \
        fprintf(stderr, KGRN "SUCCESS: %s %s:%d " KNRM S ENDL,                     \
                __extension__ __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__);    \
    }                                                                              \
    while (0)
#else
#define st_success(S, ...)
#endif

#ifdef ST_ERROR
#define st_error(S, ...)                                                           \
    do                                                                             \
    {                                                                              \
        fprintf(stderr, KRED "ERROR: %s %s:%d " KNRM S ENDL,                       \
                __extension__ __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__);    \
    }                                                                              \
    while (0)
#else
#define st_error(S, ...)
#endif

#endif /* DEBUG_H */
