// debug_macros.h
#ifndef DEBUG_MACROS_H
#define DEBUG_MACROS_H

#include <stdio.h>

extern int IsVerbose(void);

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define TRACE_FORMAT_LOCATION(message) __FILE__ ":" TOSTRING(__LINE__) " " message "\n"
#define TRACE_FORMAT_ERROR(message)    "[ERROR]   " TRACE_FORMAT_LOCATION(message)
#define TRACE_FORMAT_WARNING(message)  "[WARNING] " TRACE_FORMAT_LOCATION(message)
#define TRACE_FORMAT_INFO(message)     "[INFO]    " TRACE_FORMAT_LOCATION(message)
#define TRACE_FORMAT_SUCCESS(message)  "[SUCCESS] " TRACE_FORMAT_LOCATION(message)

#define TRACE(format, ...) \
do { \
if (IsVerbose()) { \
printf(format __VA_OPT__(,) __VA_ARGS__); \
} \
} while (0)

#define TRACE_ERROR(message, ...)   TRACE(TRACE_FORMAT_ERROR(message) __VA_OPT__(,) __VA_ARGS__)
#define TRACE_WARNING(message, ...) TRACE(TRACE_FORMAT_WARNING(message) __VA_OPT__(,) __VA_ARGS__)
#define TRACE_INFO(message, ...)    TRACE(TRACE_FORMAT_INFO(message) __VA_OPT__(,) __VA_ARGS__)
#define TRACE_SUCCESS(message, ...) TRACE(TRACE_FORMAT_SUCCESS(message) __VA_OPT__(,) __VA_ARGS__)

#define ERROR(message, ...) \
    fprintf(stderr, TRACE_FORMAT_ERROR(message) __VA_OPT__(,) __VA_ARGS__); \

#endif // DEBUG_MACROS_H